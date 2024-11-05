#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>
#include <omp.h>

#define INITIAL_CAPACITY 1024
#define LINE_BUFFER 2048
#define CHUNK_SIZE 32

// Pack edge data for better cache utilization
typedef struct {
    int to;
    int weight;
} __attribute__((packed)) CompactEdge;

typedef struct {
    CompactEdge* edges;
    int* offsets;
    int edge_count;
    int vertex_count;
} EdgeIndex;

typedef struct Edge {
    int target;
    int weight;
    struct Edge* next;
} Edge;

typedef struct {
    Edge** edges;
    int* node_ids;
    int node_count;
    int capacity;
} Graph;

typedef struct {
    int* ids;
    int* indices;
    int count;
    int capacity;
} NodeMap;

// Enhanced thread statistics
typedef struct {
    int improvements;
    int swaps_considered;
    int beneficial_swaps;
    double total_improvement_value;  // Track total value of improvements
    double max_improvement;          // Track largest improvement
    time_t last_improvement_time;    // Track time of last improvement
    char padding[64];  // Prevent false sharing
} ThreadStats;

// Thread-local improvement buffer
typedef struct {
    int vertex1;
    int vertex2;
    int delta;
    bool has_improvement;
    char padding[64]; // Prevent false sharing
} ImprovementBuffer;

void collect_neighborhood(const EdgeIndex* idx, int vertex, bool* in_neighborhood);

// Error handling
void error_exit(const char* msg) {
    fprintf(stderr, "Error: %s\n", msg);
    if (errno) perror("System error");
    exit(1);
}

// Memory management
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) error_exit("Memory allocation failed");
    return ptr;
}

void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) error_exit("Memory reallocation failed");
    return new_ptr;
}

// Time formatting helper
void format_elapsed_time(time_t seconds, char* buffer) {
    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int secs = seconds % 60;
    sprintf(buffer, "%02d:%02d:%02d", hours, minutes, secs);
}

// Progress bar helper
void print_progress_bar(double percentage, int width) {
    int filled = (int)(percentage * width / 100.0);
    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%", percentage);
}

// Node map functions
NodeMap* create_node_map(void) {
    NodeMap* map = safe_malloc(sizeof(NodeMap));
    map->capacity = INITIAL_CAPACITY;
    map->count = 0;
    map->ids = safe_malloc(sizeof(int) * INITIAL_CAPACITY);
    map->indices = safe_malloc(sizeof(int) * INITIAL_CAPACITY);
    return map;
}

void free_node_map(NodeMap* map) {
    if (map) {
        free(map->ids);
        free(map->indices);
        free(map);
    }
}

int get_node_index(NodeMap* map, int id) {
    int left = 0, right = map->count - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (map->ids[mid] == id) return map->indices[mid];
        if (map->ids[mid] < id) left = mid + 1;
        else right = mid - 1;
    }

    if (map->count >= map->capacity) {
        map->capacity *= 2;
        map->ids = safe_realloc(map->ids, sizeof(int) * map->capacity);
        map->indices = safe_realloc(map->indices, sizeof(int) * map->capacity);
    }

    int pos = left;
    memmove(&map->ids[pos + 1], &map->ids[pos], 
            (map->count - pos) * sizeof(int));
    memmove(&map->indices[pos + 1], &map->indices[pos],
            (map->count - pos) * sizeof(int));
    
    map->ids[pos] = id;
    map->indices[pos] = map->count;
    map->count++;
    return map->indices[pos];
}

// Graph functions
Graph* create_graph(void) {
    Graph* g = safe_malloc(sizeof(Graph));
    g->capacity = INITIAL_CAPACITY;
    g->node_count = 0;
    g->edges = safe_malloc(sizeof(Edge*) * INITIAL_CAPACITY);
    g->node_ids = safe_malloc(sizeof(int) * INITIAL_CAPACITY);
    memset(g->edges, 0, sizeof(Edge*) * INITIAL_CAPACITY);
    return g;
}

void free_graph(Graph* g) {
    if (g) {
        for (int i = 0; i < g->node_count; i++) {
            Edge* current = g->edges[i];
            while (current) {
                Edge* next = current->next;
                free(current);
                current = next;
            }
        }
        free(g->edges);
        free(g->node_ids);
        free(g);
    }
}

void add_edge(Graph* g, int from_idx, int to_idx, int weight) {
    while (from_idx >= g->capacity) {
        int new_capacity = g->capacity * 2;
        g->edges = safe_realloc(g->edges, sizeof(Edge*) * new_capacity);
        g->node_ids = safe_realloc(g->node_ids, sizeof(int) * new_capacity);
        memset(g->edges + g->capacity, 0, sizeof(Edge*) * (new_capacity - g->capacity));
        g->capacity = new_capacity;
    }

    if (from_idx >= g->node_count) {
        g->node_count = from_idx + 1;
    }

    Edge* edge = safe_malloc(sizeof(Edge));
    edge->target = to_idx;
    edge->weight = weight;
    edge->next = g->edges[from_idx];
    g->edges[from_idx] = edge;
}

EdgeIndex* create_edge_index(Graph* g) {
    EdgeIndex* idx = safe_malloc(sizeof(EdgeIndex));
    idx->vertex_count = g->node_count;
    
    printf("Creating edge index for %d vertices...\n", g->node_count);
    
    // Count edges and create offset array
    idx->offsets = safe_malloc(sizeof(int) * (g->node_count + 1));
    memset(idx->offsets, 0, sizeof(int) * (g->node_count + 1));
    
    // First pass: count edges per vertex
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < g->node_count; i++) {
        int count = 0;
        for (Edge* e = g->edges[i]; e; e = e->next) count++;
        idx->offsets[i + 1] = count;
        
        if (i % 10000 == 0) {
            #pragma omp critical
            {
                printf("\rCounting edges: %.1f%%", 100.0 * i / g->node_count);
                fflush(stdout);
            }
        }
    }
    printf("\rCounting edges: 100.0%%\n");
    
    // Compute cumulative offsets
    for (int i = 1; i <= g->node_count; i++) {
        idx->offsets[i] += idx->offsets[i - 1];
    }
    idx->edge_count = idx->offsets[g->node_count];
    
    printf("Allocating space for %d edges...\n", idx->edge_count);
    idx->edges = safe_malloc(sizeof(CompactEdge) * idx->edge_count);
    
    // Fill edge array
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < g->node_count; i++) {
        int offset = idx->offsets[i];
        for (Edge* e = g->edges[i]; e; e = e->next) {
            idx->edges[offset].to = e->target;
            idx->edges[offset].weight = e->weight;
            offset++;
        }
        
        if (i % 10000 == 0) {
            #pragma omp critical
            {
                printf("\rPopulating edges: %.1f%%", 100.0 * i / g->node_count);
                fflush(stdout);
            }
        }
    }
    printf("\rPopulating edges: 100.0%%\n");
    
    return idx;
}

void free_edge_index(EdgeIndex* idx) {
    if (idx) {
        free(idx->edges);
        free(idx->offsets);
        free(idx);
    }
}

inline int get_edge_weight(const EdgeIndex* idx, int from, int to) {
    const int start = idx->offsets[from];
    const int end = idx->offsets[from + 1];
    
    // Binary search
    int left = start;
    int right = end - 1;
    
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (idx->edges[mid].to == to) {
            return idx->edges[mid].weight;
        }
        if (idx->edges[mid].to < to) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return 0;
}

inline int get_edge_contribution(const EdgeIndex* male_idx, const EdgeIndex* female_idx,
                               int male_from, int male_to,
                               int female_from, int female_to) {
    int male_weight = get_edge_weight(male_idx, male_from, male_to);
    if (male_weight == 0) return 0;
    
    int female_weight = get_edge_weight(female_idx, female_from, female_to);
    if (female_weight == 0) return 0;
    
    return (male_weight < female_weight) ? male_weight : female_weight;
}

Graph* load_graph(const char* filename, NodeMap* map) {
    FILE* file = fopen(filename, "r");
    if (!file) error_exit("Cannot open graph file");

    Graph* g = create_graph();
    char line[LINE_BUFFER];

    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        free_graph(g);
        error_exit("Empty file");
    }

    printf("Loading graph from %s...\n", filename);
    int line_count = 0;
    time_t start_time = time(NULL);
    time_t last_update = start_time;
    
    while (fgets(line, sizeof(line), file)) {
        int from_id, to_id, weight;
        if (sscanf(line, "%d,%d,%d", &from_id, &to_id, &weight) != 3) {
            fprintf(stderr, "Warning: Invalid format at line %d: %s", line_count + 2, line);
            continue;
        }

        int from_idx = get_node_index(map, from_id);
        int to_idx = get_node_index(map, to_id);
        add_edge(g, from_idx, to_idx, weight);
        
        line_count++;
        if (line_count % 100000 == 0) {
            time_t current = time(NULL);
            if (current - last_update >= 1) {
                printf("\rProcessed %d edges (%.1f edges/sec)...", 
                       line_count,
                       line_count / (double)(current - start_time));
                fflush(stdout);
                last_update = current;
            }
        }
    }
    printf("\nCompleted loading %d edges\n", line_count);

    fclose(file);
    return g;
}

int* load_matching(const char* filename, NodeMap* male_map, NodeMap* female_map) {
    FILE* file = fopen(filename, "r");
    if (!file) error_exit("Cannot open matching file");

    printf("Loading matching from %s...\n", filename);
    
    int* matching = safe_malloc(sizeof(int) * male_map->count);
    for (int i = 0; i < male_map->count; i++) {
        matching[i] = -1;
    }

    char line[LINE_BUFFER];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        free(matching);
        error_exit("Empty matching file");
    }

    int line_count = 0;
    time_t start_time = time(NULL);
    time_t last_update = start_time;
    
    while (fgets(line, sizeof(line), file)) {
        int male_id, female_id;
        if (sscanf(line, "%d,%d", &male_id, &female_id) != 2) {
            fprintf(stderr, "Warning: Invalid format at line %d: %s", line_count + 2, line);
            continue;
        }
        
        int male_idx = get_node_index(male_map, male_id);
        int female_idx = get_node_index(female_map, female_id);
        matching[male_idx] = female_idx;
        line_count++;
        
        if (line_count % 10000 == 0) {
            time_t current = time(NULL);
            if (current - last_update >= 1) {
                printf("\rLoaded %d matches...", line_count);
                fflush(stdout);
                last_update = current;
            }
        }
    }

    printf("\nCompleted loading %d matches\n", line_count);
    fclose(file);
    return matching;
}

void write_matching(const char* filename, int* matching, NodeMap* male_map, NodeMap* female_map) {
    FILE* file = fopen(filename, "w");
    if (!file) error_exit("Cannot open output file");

    fprintf(file, "Male ID,Female ID\n");
    int written = 0;
    for (int i = 0; i < male_map->count; i++) {
        if (matching[i] >= 0 && matching[i] < female_map->count) {
            fprintf(file, "%d,%d\n", male_map->ids[i], female_map->ids[matching[i]]);
            written++;
        }
    }

    fclose(file);
    printf("Wrote %d matches to %s\n", written, filename);
}

int calculate_score(Graph* male_graph, Graph* female_graph, int* matching) {
    int score = 0;
    time_t start = time(NULL);
    printf("Calculating score...\n");
    
    #pragma omp parallel for reduction(+:score) schedule(dynamic, 1000)
    for (int i = 0; i < male_graph->node_count; i++) {
        int local_score = 0;
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (matching[i] < 0 || matching[e->target] < 0) continue;
            
            int female_from = matching[i];
            int female_to = matching[e->target];
            
            int female_weight = 0;
            for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                if (f->target == female_to) {
                    female_weight = f->weight;
                    break;
                }
            }
            
            local_score += (e->weight < female_weight) ? e->weight : female_weight;
        }
        score += local_score;
        
        if (i % 10000 == 0) {
            #pragma omp critical
            {
                printf("\rProgress: %.1f%%", 100.0 * i / male_graph->node_count);
                fflush(stdout);
            }
        }
    }
    
    printf("\rScore calculation complete: %d (took %ld seconds)\n", 
           score, time(NULL) - start);
    return score;
}

int calculate_swap_delta_fast(const EdgeIndex* male_idx, const EdgeIndex* female_idx,
                            const int* matching, int v1, int v2) {
    int delta = 0;
    
    // Process outgoing edges from v1
    for (int i = male_idx->offsets[v1]; i < male_idx->offsets[v1 + 1]; i++) {
        const int end = male_idx->edges[i].to;
        if (matching[end] >= 0) {
            delta -= get_edge_contribution(male_idx, female_idx,
                                         v1, end,
                                         matching[v1], matching[end]);
            delta += get_edge_contribution(male_idx, female_idx,
                                         v1, end,
                                         matching[v2], matching[end]);
        }
    }
    
    // Process outgoing edges from v2
    for (int i = male_idx->offsets[v2]; i < male_idx->offsets[v2 + 1]; i++) {
        const int end = male_idx->edges[i].to;
        if (matching[end] >= 0) {
            delta -= get_edge_contribution(male_idx, female_idx,
                                         v2, end,
                                         matching[v2], matching[end]);
            delta += get_edge_contribution(male_idx, female_idx,
                                         v2, end,
                                         matching[v1], matching[end]);
        }
    }
    
    // Process edges coming into v1 and v2
    #pragma omp parallel for schedule(static, CHUNK_SIZE) reduction(+:delta)
    for (int from = 0; from < male_idx->vertex_count; from++) {
        if (from == v1 || from == v2) continue;
        
        bool found_v1 = false, found_v2 = false;
        int old_v1_contrib = 0, old_v2_contrib = 0;
        
        // Check for edges to v1 and v2
        for (int i = male_idx->offsets[from]; i < male_idx->offsets[from + 1] && !(found_v1 && found_v2); i++) {
            const int to = male_idx->edges[i].to;
            if (to == v1) {
                found_v1 = true;
                old_v1_contrib = get_edge_contribution(male_idx, female_idx,
                                                     from, v1,
                                                     matching[from], matching[v1]);
            } else if (to == v2) {
                found_v2 = true;
                old_v2_contrib = get_edge_contribution(male_idx, female_idx,
                                                     from, v2,
                                                     matching[from], matching[v2]);
            }
        }
        
        if (found_v1) {
            delta -= old_v1_contrib;
            delta += get_edge_contribution(male_idx, female_idx,
                                         from, v1,
                                         matching[from], matching[v2]);
        }
        if (found_v2) {
            delta -= old_v2_contrib;
            delta += get_edge_contribution(male_idx, female_idx,
                                         from, v2,
                                         matching[from], matching[v1]);
        }
    }
    
    return delta;
}

void print_thread_stats(ThreadStats* stats, int num_threads, time_t start_time, 
                       int current_score, int initial_score) {
    time_t current = time(NULL);
    int total_improvements = 0;
    int total_swaps = 0;
    int total_beneficial = 0;
    double total_improvement_value = 0;
    double max_improvement = 0;
    
    printf("\n=== Thread Statistics ===\n");
    printf("ID  Improvements  Swaps     Success%%   Avg Gain  Last Improv\n");
    printf("--------------------------------------------------------\n");
    
    for (int i = 0; i < num_threads; i++) {
        total_improvements += stats[i].improvements;
        total_swaps += stats[i].swaps_considered;
        total_beneficial += stats[i].beneficial_swaps;
        total_improvement_value += stats[i].total_improvement_value;
        if (stats[i].max_improvement > max_improvement) 
            max_improvement = stats[i].max_improvement;
        
        double success_rate = stats[i].swaps_considered > 0 ? 
            100.0 * stats[i].beneficial_swaps / stats[i].swaps_considered : 0;
        double avg_gain = stats[i].beneficial_swaps > 0 ?
            stats[i].total_improvement_value / stats[i].beneficial_swaps : 0;
        
        char elapsed[32];
        if (stats[i].last_improvement_time > 0) {
            format_elapsed_time(current - stats[i].last_improvement_time, elapsed);
        } else {
            strcpy(elapsed, "never");
        }
        
        printf("%2d  %11d  %9d  %8.2f%%  %8.1f  %s\n",
               i, stats[i].improvements, stats[i].swaps_considered,
               success_rate, avg_gain, elapsed);
    }
    
    printf("\n=== Overall Progress ===\n");
    printf("Current score: %d (%+.2f%% from initial %d)\n",
           current_score, 100.0 * (current_score - initial_score) / initial_score,
           initial_score);
    printf("Total improvements: %d (%.1f per hour)\n",
           total_improvements,
           3600.0 * total_improvements / (current - start_time));
    printf("Success rate: %.2f%% (%d beneficial out of %d tried)\n",
           100.0 * total_beneficial / (total_swaps > 0 ? total_swaps : 1),
           total_beneficial, total_swaps);
    printf("Average improvement: %.1f points (max: %.1f)\n",
           total_beneficial > 0 ? total_improvement_value / total_beneficial : 0,
           max_improvement);
    printf("Processing rate: %.1f swaps/sec\n",
           total_swaps / (double)(current - start_time));
}

// Add this function before optimize_matching
void collect_neighborhood(const EdgeIndex* idx, int vertex, bool* in_neighborhood) {
    memset(in_neighborhood, 0, idx->vertex_count * sizeof(bool));
    in_neighborhood[vertex] = true;
    
    // Mark outgoing neighbors and their neighbors
    for (int i = idx->offsets[vertex]; i < idx->offsets[vertex + 1]; i++) {
        const int target = idx->edges[i].to;
        in_neighborhood[target] = true;
        
        // Mark second hop neighbors
        for (int j = idx->offsets[target]; j < idx->offsets[target + 1]; j++) {
            in_neighborhood[idx->edges[j].to] = true;
        }
    }
    
    // Mark incoming neighbors and their neighbors
    #pragma omp parallel for schedule(static, CHUNK_SIZE)
    for (int from = 0; from < idx->vertex_count; from++) {
        bool found_edge = false;
        
        // Check for edge to vertex
        for (int i = idx->offsets[from]; i < idx->offsets[from + 1]; i++) {
            if (idx->edges[i].to == vertex) {
                found_edge = true;
                #pragma omp atomic write
                in_neighborhood[from] = true;
                break;
            }
        }
        
        // If there's an edge to vertex, mark vertices that can reach 'from'
        if (found_edge) {
            for (int i = 0; i < idx->vertex_count; i++) {
                for (int j = idx->offsets[i]; j < idx->offsets[i + 1]; j++) {
                    if (idx->edges[j].to == from) {
                        #pragma omp atomic write
                        in_neighborhood[i] = true;
                        break;
                    }
                }
            }
        }
    }
}

void optimize_matching(Graph* male_graph, Graph* female_graph, int* matching,
                      NodeMap* male_map, NodeMap* female_map) {
    printf("\n=== Starting Enhanced OpenMP Graph Matching Optimization ===\n");
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            printf("Using %d threads\n", num_threads);
        }
    }
    
    time_t start_time = time(NULL);
    time_t last_update = start_time;
    
    printf("Building edge indices...\n");
    EdgeIndex* male_idx = create_edge_index(male_graph);
    EdgeIndex* female_idx = create_edge_index(female_graph);
    
    const int initial_score = calculate_score(male_graph, female_graph, matching);
    int current_score = initial_score;
    printf("\nInitial matching score: %d\n", initial_score);
    
    // Thread-local statistics
    ThreadStats* thread_stats = safe_malloc(num_threads * sizeof(ThreadStats));
    memset(thread_stats, 0, num_threads * sizeof(ThreadStats));
    
    // Thread-local improvement buffers
    ImprovementBuffer* improvements = safe_malloc(num_threads * sizeof(ImprovementBuffer));
    
    // Thread seeds for random number generation
    unsigned int* seeds = safe_malloc(num_threads * sizeof(unsigned int));
    for (int i = 0; i < num_threads; i++) {
        seeds[i] = (unsigned int)time(NULL) + i;
    }
    
    printf("\nStarting optimization...\n");
    
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        bool* local_in_neighborhood = safe_malloc(male_idx->vertex_count * sizeof(bool));
        int* local_neighborhood = safe_malloc(male_idx->vertex_count * sizeof(int));
        
        while (1) {
            improvements[thread_id].has_improvement = false;
            improvements[thread_id].delta = 0;
            
            for (int local_iter = 0; local_iter < 10; local_iter++) {
                const int vertex = rand_r(&seeds[thread_id]) % male_idx->vertex_count;
                
                // Collect and process neighborhood
                collect_neighborhood(male_idx, vertex, local_in_neighborhood);
                
                int neighborhood_size = 0;
                for (int i = 0; i < male_idx->vertex_count; i++) {
                    if (local_in_neighborhood[i]) {
                        local_neighborhood[neighborhood_size++] = i;
                    }
                }
                
                const int swaps_to_try = (neighborhood_size < 1000) ? neighborhood_size : 1000;
                
                for (int tried = 0; tried < swaps_to_try; tried++) {
                    const int idx = tried + (rand_r(&seeds[thread_id]) % (neighborhood_size - tried));
                    const int target = local_neighborhood[idx];
                    local_neighborhood[idx] = local_neighborhood[tried];
                    
                    if (target != vertex) {
                        thread_stats[thread_id].swaps_considered++;
                        
                        const int delta = calculate_swap_delta_fast(male_idx, female_idx,
                                                                  matching, vertex, target);
                        
                        if (delta > improvements[thread_id].delta) {
                            improvements[thread_id].delta = delta;
                            improvements[thread_id].vertex1 = vertex;
                            improvements[thread_id].vertex2 = target;
                            improvements[thread_id].has_improvement = true;
                        }
                    }
                }
            }
            
            #pragma omp barrier
            #pragma omp single
            {
                int best_thread = -1;
                int best_delta = 0;
                
                for (int i = 0; i < num_threads; i++) {
                    if (improvements[i].has_improvement && improvements[i].delta > best_delta) {
                        best_delta = improvements[i].delta;
                        best_thread = i;
                    }
                }
                
                if (best_thread >= 0) {
                    const int v1 = improvements[best_thread].vertex1;
                    const int v2 = improvements[best_thread].vertex2;
                    
                    // Perform swap
                    const int temp = matching[v1];
                    matching[v1] = matching[v2];
                    matching[v2] = temp;
                    
                    current_score += best_delta;
                    
                    // Update thread statistics
                    thread_stats[best_thread].improvements++;
                    thread_stats[best_thread].beneficial_swaps++;
                    thread_stats[best_thread].total_improvement_value += best_delta;
                    thread_stats[best_thread].last_improvement_time = time(NULL);
                    if (best_delta > thread_stats[best_thread].max_improvement)
                        thread_stats[best_thread].max_improvement = best_delta;
                    
                    char filename[256];
                    snprintf(filename, sizeof(filename), "matching_score_%d.csv",
                            current_score);
                    write_matching(filename, matching, male_map, female_map);
                    
                    time_t current = time(NULL);
                    printf("\n[%ld sec] Thread %d found improvement!\n",
                           current - start_time, best_thread);
                    printf("- Swapped %d <-> %d for +%d points\n",
                           male_map->ids[v1], male_map->ids[v2], best_delta);
                    printf("- New score: %d (%+.2f%%)\n",
                           current_score,
                           100.0 * (current_score - initial_score) / initial_score);
                }
                
                // Progress update
                time_t current = time(NULL);
                if (current - last_update >= 5) {
                    print_thread_stats(thread_stats, num_threads, start_time,
                                     current_score, initial_score);
                    last_update = current;
                }
            }
        }
        
        free(local_in_neighborhood);
        free(local_neighborhood);
    }
    
    free(seeds);
    free(improvements);
    free(thread_stats);
    free_edge_index(male_idx);
    free_edge_index(female_idx);
}



int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <male_graph.csv> <female_graph.csv> "
                "<initial_matching.csv> <output_matching.csv>\n", argv[0]);
        return 1;
    }

    NodeMap* male_map = NULL;
    NodeMap* female_map = NULL;
    Graph* male_graph = NULL;
    Graph* female_graph = NULL;
    int* matching = NULL;
    
    printf("Starting graph matching optimization\n");
    printf("------------------------------------\n");
    
    // Create node maps
    male_map = create_node_map();
    female_map = create_node_map();
    if (!male_map || !female_map) 
        error_exit("Failed to create node maps");
    
    // Load graphs and matching
    male_graph = load_graph(argv[1], male_map);
    female_graph = load_graph(argv[2], female_map);
    if (!male_graph || !female_graph)
        error_exit("Failed to load graphs");
    
    matching = load_matching(argv[3], male_map, female_map);
    if (!matching)
        error_exit("Failed to load initial matching");
    
    // Optimize matching
    optimize_matching(male_graph, female_graph, matching, male_map, female_map);
    
    // Write final result
    write_matching(argv[4], matching, male_map, female_map);
    
    // Cleanup
    printf("Cleaning up...\n");
    free(matching);
    free_graph(male_graph);
    free_graph(female_graph);
    free_node_map(male_map);
    free_node_map(female_map);
    
    printf("Done.\n");
    return 0;
}
