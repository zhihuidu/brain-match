#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>

#define MAX_ITER 10000
#define INITIAL_TEMP 100.0
#define COOLING_RATE 0.99
#define MIN_TEMP 0.01
#define LINE_BUFFER_SIZE 1024
#define NUM_THREADS 8
#define EARLY_STOP_WINDOW 1000
#define EARLY_STOP_THRESHOLD 0.001

// Data structures
typedef struct {
    int target;
    int weight;
} Edge;

typedef struct {
    Edge *edges;
    int *next;
    int size;
    int capacity;
} EdgeHashTable;

typedef struct {
    EdgeHashTable *adj;
    int num_nodes;
} Graph;

typedef struct {
    int *id_to_index;
    int *index_to_id;
    int max_id;
    int count;
} NodeMapping;

typedef struct {
    Graph *g1;
    Graph *g2;
    int *matching;
    int *best_matching;
    int thread_id;
    int num_threads;
    int num_nodes;
    volatile int *best_score;
    pthread_mutex_t *score_mutex;
} ThreadWork;

// Error handling
void handle_error(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    if (errno != 0) {
        fprintf(stderr, "System error: %s\n", strerror(errno));
    }
    exit(EXIT_FAILURE);
}

// Memory management
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) handle_error("Memory allocation failed");
    return ptr;
}

void *safe_realloc(void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr) handle_error("Memory reallocation failed");
    return new_ptr;
}

// Edge hash table functions
EdgeHashTable create_edge_hash_table(int capacity) {
    EdgeHashTable table;
    table.capacity = capacity;
    table.size = 0;
    table.edges = safe_malloc(capacity * sizeof(Edge));
    table.next = safe_malloc(capacity * sizeof(int));
    memset(table.next, -1, capacity * sizeof(int));
    return table;
}

void free_edge_hash_table(EdgeHashTable *table) {
    if (table) {
        free(table->edges);
        free(table->next);
    }
}

int edge_hash(int target, int capacity) {
    return target % capacity;
}

void insert_edge(EdgeHashTable *table, int target, int weight) {
    int hash = edge_hash(target, table->capacity);
    
    int curr = hash;
    while (table->next[curr] != -1) {
        if (table->edges[curr].target == target) {
            table->edges[curr].weight = weight;
            return;
        }
        curr = table->next[curr];
    }
    
    table->edges[table->size] = (Edge){target, weight};
    table->next[table->size] = table->next[hash];
    table->next[hash] = table->size;
    table->size++;
    
    if (table->size >= table->capacity * 0.75) {
        int old_capacity = table->capacity;
        table->capacity *= 2;
        table->edges = safe_realloc(table->edges, table->capacity * sizeof(Edge));
        table->next = safe_realloc(table->next, table->capacity * sizeof(int));
        memset(table->next + old_capacity, -1, (table->capacity - old_capacity) * sizeof(int));
    }
}

Edge* find_edge(EdgeHashTable *table, int target) {
    int hash = edge_hash(target, table->capacity);
    int curr = hash;
    while (curr != -1) {
        if (table->edges[curr].target == target) {
            return &table->edges[curr];
        }
        curr = table->next[curr];
    }
    return NULL;
}

// Graph functions
Graph *create_graph(int num_nodes) {
    Graph *graph = safe_malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->adj = safe_malloc(num_nodes * sizeof(EdgeHashTable));
    
    for (int i = 0; i < num_nodes; i++) {
        graph->adj[i] = create_edge_hash_table(16);  // Start small, will grow as needed
    }
    
    return graph;
}

void free_graph(Graph *graph) {
    if (graph) {
        for (int i = 0; i < graph->num_nodes; i++) {
            free_edge_hash_table(&graph->adj[i]);
        }
        free(graph->adj);
        free(graph);
    }
}

// Node mapping functions
NodeMapping *create_node_mapping(int initial_size) {
    NodeMapping *mapping = safe_malloc(sizeof(NodeMapping));
    mapping->id_to_index = safe_malloc(initial_size * sizeof(int));
    mapping->index_to_id = safe_malloc(initial_size * sizeof(int));
    memset(mapping->id_to_index, -1, initial_size * sizeof(int));
    mapping->max_id = initial_size - 1;
    mapping->count = 0;
    return mapping;
}

void free_node_mapping(NodeMapping *mapping) {
    if (mapping) {
        free(mapping->id_to_index);
        free(mapping->index_to_id);
        free(mapping);
    }
}

void expand_mapping(NodeMapping *mapping, int new_max_id) {
    int new_size = new_max_id + 1;
    mapping->id_to_index = safe_realloc(mapping->id_to_index, new_size * sizeof(int));
    mapping->index_to_id = safe_realloc(mapping->index_to_id, new_size * sizeof(int));
    
    for (int i = mapping->max_id + 1; i < new_size; i++) {
        mapping->id_to_index[i] = -1;
    }
    
    mapping->max_id = new_max_id;
}

int add_node_to_mapping(NodeMapping *mapping, int node_id) {
    if (node_id > mapping->max_id) {
        expand_mapping(mapping, node_id * 2);
    }
    
    if (mapping->id_to_index[node_id] == -1) {
        int new_index = mapping->count;
        mapping->id_to_index[node_id] = new_index;
        mapping->index_to_id[new_index] = node_id;
        mapping->count++;
        return new_index;
    }
    
    return mapping->id_to_index[node_id];
}

// Score calculation functions
int calculate_edge_alignment(Graph *g1, Graph *g2, int node1, int match1,
                           int target1, int match_target) {
    Edge *edge1 = find_edge(&g1->adj[node1], target1);
    Edge *edge2 = find_edge(&g2->adj[match1], match_target);
    
    if (!edge1) return 0;  // No edge in male graph
    if (!edge2) return 0;  // No edge in female graph
    
    // Return minimum of the two edge weights
    return edge1->weight < edge2->weight ? edge1->weight : edge2->weight;
}

int calculate_total_score(Graph *g1, Graph *g2, int *matching) {
    int total_score = 0;
    
    for (int i = 0; i < g1->num_nodes; i++) {
        if (matching[i] < 0 || matching[i] >= g2->num_nodes) continue;
        
        EdgeHashTable *adj = &g1->adj[i];
        for (int j = 0; j < adj->size; j++) {
            Edge *edge = &adj->edges[j];
            int target = edge->target;
            
            if (target < 0 || target >= g1->num_nodes ||
                matching[target] < 0 || matching[target] >= g2->num_nodes) continue;
            
            total_score += calculate_edge_alignment(g1, g2, i, matching[i],
                                                 target, matching[target]);
        }
    }
    
    return total_score;
}

int calculate_swap_delta(Graph *g1, Graph *g2, int *matching,
                        int node1, int node2) {
    int before_score = 0;
    int after_score = 0;
    
    EdgeHashTable *adj1 = &g1->adj[node1];
    EdgeHashTable *adj2 = &g1->adj[node2];
    
    for (int i = 0; i < adj1->size; i++) {
        Edge *edge = &adj1->edges[i];
        int target = edge->target;
        if (target != node2) {
            before_score += calculate_edge_alignment(g1, g2, node1, matching[node1],
                                                  target, matching[target]);
        }
    }
    
    for (int i = 0; i < adj2->size; i++) {
        Edge *edge = &adj2->edges[i];
        int target = edge->target;
        if (target != node1) {
            before_score += calculate_edge_alignment(g1, g2, node2, matching[node2],
                                                  target, matching[target]);
        }
    }

    int temp = matching[node1];
    matching[node1] = matching[node2];
    matching[node2] = temp;

    for (int i = 0; i < adj1->size; i++) {
        Edge *edge = &adj1->edges[i];
        int target = edge->target;
        if (target != node2) {
            after_score += calculate_edge_alignment(g1, g2, node1, matching[node1],
                                                 target, matching[target]);
        }
    }
    
    for (int i = 0; i < adj2->size; i++) {
        Edge *edge = &adj2->edges[i];
        int target = edge->target;
        if (target != node1) {
            after_score += calculate_edge_alignment(g1, g2, node2, matching[node2],
                                                 target, matching[target]);
        }
    }

    matching[node2] = matching[node1];
    matching[node1] = temp;

    return after_score - before_score;
}

// Thread function for parallel optimization
void* optimize_thread(void *arg) {
    ThreadWork *work = (ThreadWork*)arg;
    int *local_matching = safe_malloc(work->num_nodes * sizeof(int));
    memcpy(local_matching, work->matching, work->num_nodes * sizeof(int));
    
    int current_score = calculate_total_score(work->g1, work->g2, local_matching);
    double temperature = INITIAL_TEMP;
    
    int window_scores[EARLY_STOP_WINDOW];
    int window_pos = 0;
    bool window_full = false;
    
    srand(time(NULL) + work->thread_id);
    
    for (int iter = work->thread_id; iter < MAX_ITER; iter += work->num_threads) {
        if (temperature < MIN_TEMP) break;
        
        int node1 = rand() % work->num_nodes;
        int node2 = rand() % work->num_nodes;
        if (node1 == node2) continue;
        
        int delta = calculate_swap_delta(work->g1, work->g2, local_matching,
                                      node1, node2);
        
        if (delta > 0 || (exp(delta / temperature) > (double)rand() / RAND_MAX)) {
            int temp = local_matching[node1];
            local_matching[node1] = local_matching[node2];
            local_matching[node2] = temp;
            current_score += delta;
            
            if (current_score > *work->best_score) {
                pthread_mutex_lock(work->score_mutex);
                if (current_score > *work->best_score) {
                    *work->best_score = current_score;
                    memcpy(work->best_matching, local_matching,
                           work->num_nodes * sizeof(int));
                }
                pthread_mutex_unlock(work->score_mutex);
            }
        }
        
        window_scores[window_pos] = current_score;
        window_pos = (window_pos + 1) % EARLY_STOP_WINDOW;
        if (window_pos == 0) window_full = true;
        
        if (window_full) {
            double avg = 0;
            double var = 0;
            for (int i = 0; i < EARLY_STOP_WINDOW; i++) {
                avg += window_scores[i];
            }
            avg /= EARLY_STOP_WINDOW;
            
            for (int i = 0; i < EARLY_STOP_WINDOW; i++) {
                double diff = window_scores[i] - avg;
                var += diff * diff;
            }
            var /= EARLY_STOP_WINDOW;
            
            if (var < EARLY_STOP_THRESHOLD * avg) {
                break;
            }
        }
        
        temperature *= COOLING_RATE;
    }
    
    free(local_matching);
    return NULL;
}

// Parallel optimization function
void optimize_matching_parallel(Graph *g1, Graph *g2, int *matching) {
    int num_nodes = g1->num_nodes;
    int best_score = calculate_total_score(g1, g2, matching);
    int *best_matching = safe_malloc(num_nodes * sizeof(int));
    memcpy(best_matching, matching, num_nodes * sizeof(int));
    
    pthread_t threads[NUM_THREADS];
    ThreadWork thread_work[NUM_THREADS];
    pthread_mutex_t score_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_work[i] = (ThreadWork){
            .g1 = g1,
            .g2 = g2,
            .matching = matching,
            .best_matching = best_matching,
            .thread_id = i,
            .num_threads = NUM_THREADS,
            .num_nodes = num_nodes,
            .best_score = &best_score,
            .score_mutex = &score_mutex
        };
        pthread_create(&threads[i], NULL, optimize_thread, &thread_work[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    memcpy(matching, best_matching, num_nodes * sizeof(int));
    free(best_matching);
}

// File loading functions
Graph *load_graph(const char *filename, NodeMapping *mapping) {
    FILE *file = fopen(filename, "r");
    if (!file) handle_error("Cannot open graph file");

    char line[LINE_BUFFER_SIZE];
    if (!fgets(line, sizeof(line), file)) {  // Skip header
        fclose(file);
        handle_error("Empty graph file");
    }

    Graph *graph = create_graph(1000);  // Start with reasonable size
    long line_number = 1;

    while (fgets(line, sizeof(line), file)) {
        line_number++;
        int src_id, tgt_id, weight;
        
        if (sscanf(line, "%d,%d,%d", &src_id, &tgt_id, &weight) != 3) {
            fprintf(stderr, "Warning: Invalid format at line %ld\n", line_number);
            continue;
        }

        int src_idx = add_node_to_mapping(mapping, src_id);
        int tgt_idx = add_node_to_mapping(mapping, tgt_id);

        // Ensure graph has enough capacity
        if (src_idx >= graph->num_nodes || tgt_idx >= graph->num_nodes) {
            int new_size = (src_idx > tgt_idx ? src_idx : tgt_idx) + 1;
            Graph *new_graph = create_graph(new_size);
            
            // Copy existing edges
            for (int i = 0; i < graph->num_nodes; i++) {
                EdgeHashTable *adj = &graph->adj[i];
                for (int j = 0; j < adj->size; j++) {
                    Edge *edge = &adj->edges[j];
                    insert_edge(&new_graph->adj[i], edge->target, edge->weight);
                }
            }
            
            free_graph(graph);
            graph = new_graph;
        }

        insert_edge(&graph->adj[src_idx], tgt_idx, weight);
    }

    fclose(file);
    return graph;
}

int *load_initial_matching(const char *filename, NodeMapping *map_g1, NodeMapping *map_g2) {
    FILE *file = fopen(filename, "r");
    if (!file) handle_error("Cannot open initial matching file");

    int *matching = safe_malloc(map_g1->count * sizeof(int));
    for (int i = 0; i < map_g1->count; i++) {
        matching[i] = -1;
    }

    char line[LINE_BUFFER_SIZE];
    if (!fgets(line, sizeof(line), file)) {  // Skip header
        fclose(file);
        free(matching);
        handle_error("Empty matching file");
    }

    while (fgets(line, sizeof(line), file)) {
        int node_g1_id, node_g2_id;
        
        if (sscanf(line, "%d,%d", &node_g1_id, &node_g2_id) != 2) {
            continue;
        }

        if (node_g1_id > map_g1->max_id || node_g2_id > map_g2->max_id) {
            continue;
        }

        int g1_idx = map_g1->id_to_index[node_g1_id];
        int g2_idx = map_g2->id_to_index[node_g2_id];

        if (g1_idx >= 0 && g2_idx >= 0) {
            matching[g1_idx] = g2_idx;
        }
    }

    fclose(file);
    return matching;
}

void write_matching(const char *filename, int *matching, NodeMapping *map_g1, NodeMapping *map_g2) {
    FILE *file = fopen(filename, "w");
    if (!file) handle_error("Cannot open output file for writing");

    fprintf(file, "Male ID,Female ID\n");
    for (int i = 0; i < map_g1->count; i++) {
        if (matching[i] >= 0 && matching[i] < map_g2->count) {
            fprintf(file, "%d,%d\n", map_g1->index_to_id[i], map_g2->index_to_id[matching[i]]);
        }
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <initial_matching.csv> <output_matching.csv>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize random seed
    srand(time(NULL));

    // Create node mappings
    NodeMapping *map_g1 = create_node_mapping(1000);
    NodeMapping *map_g2 = create_node_mapping(1000);
    
    // Load graphs
    printf("Loading graph 1...\n");
    Graph *g1 = load_graph(argv[1], map_g1);
    
    printf("Loading graph 2...\n");
    Graph *g2 = load_graph(argv[2], map_g2);

    if (map_g1->count != map_g2->count) {
        fprintf(stderr, "Error: Graphs have different numbers of nodes (G1: %d, G2: %d)\n",
                map_g1->count, map_g2->count);
        free_graph(g1);
        free_graph(g2);
        free_node_mapping(map_g1);
        free_node_mapping(map_g2);
        return EXIT_FAILURE;
    }

    // Load initial matching
    printf("Loading initial matching...\n");
    int *matching = load_initial_matching(argv[3], map_g1, map_g2);

    // Calculate and display initial score
    int initial_score = calculate_total_score(g1, g2, matching);
    printf("Initial matching score: %d\n", initial_score);

    // Optimize matching
    printf("Optimizing matching using %d threads...\n", NUM_THREADS);
    optimize_matching_parallel(g1, g2, matching);

    // Calculate and display final score
    int final_score = calculate_total_score(g1, g2, matching);
    printf("Final matching score: %d\n", final_score);
    
    if (initial_score > 0) {
        double improvement = 100.0 * (final_score - initial_score) / initial_score;
        printf("Improvement: %.2f%%\n", improvement);
    }

    // Write results
    printf("Writing results to %s...\n", argv[4]);
    write_matching(argv[4], matching, map_g1, map_g2);

    // Cleanup
    printf("Cleaning up...\n");
    free(matching);
    free_graph(g1);
    free_graph(g2);
    free_node_mapping(map_g1);
    free_node_mapping(map_g2);

    printf("Done.\n");
    return EXIT_SUCCESS;
}
