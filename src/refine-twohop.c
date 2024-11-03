#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>

#define INITIAL_CAPACITY 1024
#define LINE_BUFFER 2048

// Data structures
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

// NodeMap functions
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
    // Binary search for existing ID
    int left = 0, right = map->count - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (map->ids[mid] == id) return map->indices[mid];
        if (map->ids[mid] < id) left = mid + 1;
        else right = mid - 1;
    }

    // Add new ID if not found
    if (map->count >= map->capacity) {
        map->capacity *= 2;
        map->ids = safe_realloc(map->ids, sizeof(int) * map->capacity);
        map->indices = safe_realloc(map->indices, sizeof(int) * map->capacity);
    }

    // Insert maintaining sorted order
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
        if (line_count % 1000000 == 0) {
            printf("Processed %d edges...\n", line_count);
        }
    }

    fclose(file);
    printf("Loaded %d nodes and %d edges\n", map->count, line_count);
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
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        free(matching);
        error_exit("Empty matching file");
    }

    int line_count = 0;
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
    }

    fclose(file);
    printf("Loaded %d matches\n", line_count);
    return matching;
}

void write_matching(const char* filename, int* matching, NodeMap* male_map, NodeMap* female_map) {
    FILE* file = fopen(filename, "w");
    if (!file) error_exit("Cannot open output file");

    fprintf(file, "Male ID,Female ID\n");
    for (int i = 0; i < male_map->count; i++) {
        if (matching[i] >= 0 && matching[i] < female_map->count) {
            fprintf(file, "%d,%d\n", male_map->ids[i], female_map->ids[matching[i]]);
        }
    }

    fclose(file);
    printf("Wrote matching to %s\n", filename);
}

int calculate_score(Graph* male_graph, Graph* female_graph, int* matching) {
    int score = 0;
    for (int i = 0; i < male_graph->node_count; i++) {
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (matching[i] < 0 || matching[e->target] < 0) continue;
            
            int female_from = matching[i];
            int female_to = matching[e->target];
            
            // Find corresponding edge in female graph
            int female_weight = 0;
            for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                if (f->target == female_to) {
                    female_weight = f->weight;
                    break;
                }
            }
            
            score += (e->weight < female_weight) ? e->weight : female_weight;
        }
    }
    return score;
}

long long calculate_node_weight(Graph* graph, int node) {
    long long weight_sum = 0;
    
    // Add weights of outgoing edges
    for (Edge* e = graph->edges[node]; e; e = e->next) {
        weight_sum += e->weight;
        for (Edge* e2 = graph->edges[e->target]; e2; e2 = e2->next) {
            weight_sum += e2->weight;
        }
    }
    
    // Add weights of incoming edges
    for (int i = 0; i < graph->node_count; i++) {
        for (Edge* e = graph->edges[i]; e; e = e->next) {
            if (e->target == node) {
                weight_sum += e->weight;
                for (Edge* e2 = graph->edges[i]; e2; e2 = e2->next) {
                    weight_sum += e2->weight;
                }
                break;
            }
        }
    }
    
    return weight_sum;
}

// Calculate score change for a potential swap
int calculate_swap_delta(Graph* male_graph, Graph* female_graph, int* matching, 
                        int vertex1, int vertex2) {
    int delta = 0;
    
    // Remove old edges' contributions
    // First vertex outgoing edges
    for (Edge* e = male_graph->edges[vertex1]; e; e = e->next) {
        int female_from = matching[vertex1];
        int female_to = matching[e->target];
        
        // Find old female edge weight
        int old_female_weight = 0;
        for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
            if (f->target == female_to) {
                old_female_weight = f->weight;
                break;
            }
        }
        delta -= (e->weight < old_female_weight) ? e->weight : old_female_weight;
    }
    
    // Second vertex outgoing edges
    for (Edge* e = male_graph->edges[vertex2]; e; e = e->next) {
        int female_from = matching[vertex2];
        int female_to = matching[e->target];
        
        int old_female_weight = 0;
        for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
            if (f->target == female_to) {
                old_female_weight = f->weight;
                break;
            }
        }
        delta -= (e->weight < old_female_weight) ? e->weight : old_female_weight;
    }
    
    // First vertex incoming edges
    for (int i = 0; i < male_graph->node_count; i++) {
        if (i == vertex1 || i == vertex2) continue;
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (e->target == vertex1) {
                int female_from = matching[i];
                int female_to = matching[vertex1];
                
                int old_female_weight = 0;
                for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                    if (f->target == female_to) {
                        old_female_weight = f->weight;
                        break;
                    }
                }
                delta -= (e->weight < old_female_weight) ? e->weight : old_female_weight;
            }
        }
    }
    
    // Second vertex incoming edges
    for (int i = 0; i < male_graph->node_count; i++) {
        if (i == vertex1 || i == vertex2) continue;
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (e->target == vertex2) {
                int female_from = matching[i];
                int female_to = matching[vertex2];
                
                int old_female_weight = 0;
                for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                    if (f->target == female_to) {
                        old_female_weight = f->weight;
                        break;
                    }
                }
                delta -= (e->weight < old_female_weight) ? e->weight : old_female_weight;
            }
        }
    }
    
    // Temporarily perform swap
    int temp = matching[vertex1];
    matching[vertex1] = matching[vertex2];
    matching[vertex2] = temp;
    
    // Add new edges' contributions
    // First vertex outgoing edges
    for (Edge* e = male_graph->edges[vertex1]; e; e = e->next) {
        int female_from = matching[vertex1];
        int female_to = matching[e->target];
        
        int new_female_weight = 0;
        for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
            if (f->target == female_to) {
                new_female_weight = f->weight;
                break;
            }
        }
        delta += (e->weight < new_female_weight) ? e->weight : new_female_weight;
    }
    
    // Second vertex outgoing edges
    for (Edge* e = male_graph->edges[vertex2]; e; e = e->next) {
        int female_from = matching[vertex2];
        int female_to = matching[e->target];
        
        int new_female_weight = 0;
        for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
            if (f->target == female_to) {
                new_female_weight = f->weight;
                break;
            }
        }
        delta += (e->weight < new_female_weight) ? e->weight : new_female_weight;
    }
    
    // First vertex incoming edges
    for (int i = 0; i < male_graph->node_count; i++) {
        if (i == vertex1 || i == vertex2) continue;
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (e->target == vertex1) {
                int female_from = matching[i];
                int female_to = matching[vertex1];
                
                int new_female_weight = 0;
                for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                    if (f->target == female_to) {
                        new_female_weight = f->weight;
                        break;
                    }
                }
                delta += (e->weight < new_female_weight) ? e->weight : new_female_weight;
            }
        }
    }
    
    // Second vertex incoming edges
    for (int i = 0; i < male_graph->node_count; i++) {
        if (i == vertex1 || i == vertex2) continue;
        for (Edge* e = male_graph->edges[i]; e; e = e->next) {
            if (e->target == vertex2) {
                int female_from = matching[i];
                int female_to = matching[vertex2];
                
                int new_female_weight = 0;
                for (Edge* f = female_graph->edges[female_from]; f; f = f->next) {
                    if (f->target == female_to) {
                        new_female_weight = f->weight;
                        break;
                    }
                }
                delta += (e->weight < new_female_weight) ? e->weight : new_female_weight;
            }
        }
    }
    
    // Restore original matching
    matching[vertex2] = matching[vertex1];
    matching[vertex1] = temp;
    
    return delta;
}

void optimize_matching(Graph* male_graph, Graph* female_graph, int* matching, 
                      NodeMap* male_map, NodeMap* female_map) {
    printf("\nStarting optimization...\n");
    time_t start_time = time(NULL);
    
    // Store initial score
    const int initial_score = calculate_score(male_graph, female_graph, matching);
    int current_score = initial_score;
    printf("Initial score: %d\n", initial_score);
    
    // Calculate vertex weights
    printf("Calculating vertex weights...\n");
    typedef struct {
        int vertex;
        long long weight;
    } VertexWeight;
    
    VertexWeight* vertices = safe_malloc(sizeof(VertexWeight) * male_graph->node_count);
    for (int i = 0; i < male_graph->node_count; i++) {
        vertices[i].vertex = i;
        vertices[i].weight = calculate_node_weight(male_graph, i);
        if (i % 1000 == 0) {
            printf("Processed %d vertices...\n", i);
        }
    }
    
    // Sort by weight (descending)
    printf("Sorting vertices by weight...\n");
    for (int i = 0; i < male_graph->node_count - 1; i++) {
        for (int j = i + 1; j < male_graph->node_count; j++) {
            if (vertices[i].weight < vertices[j].weight) {
                VertexWeight temp = vertices[i];
                vertices[i] = vertices[j];
                vertices[j] = temp;
            }
        }
        if (i % 1000 == 0) {
            printf("Sorted %d vertices...\n", i);
        }
    }
    
    printf("Starting refinement...\n");
    int improvements = 0;
    int iterations = 0;
    bool improved;
    
    do {
        improved = false;
        iterations++;
        
        for (int idx = 0; idx < male_graph->node_count; idx++) {
            int vertex = vertices[idx].vertex;
            bool vertex_improved = false;
            
            // Try swaps in 2-hop neighborhood
            for (Edge* e1 = male_graph->edges[vertex]; e1 && !vertex_improved; e1 = e1->next) {
                for (Edge* e2 = male_graph->edges[e1->target]; e2; e2 = e2->next) {
                    int target = e2->target;
                    if (target == vertex) continue;
                    
                    int delta = calculate_swap_delta(male_graph, female_graph, matching, 
                                                   vertex, target);
                    
                    if (delta > 0) {
                        // Perform swap
                        int temp = matching[vertex];
                        matching[vertex] = matching[target];
                        matching[target] = temp;
                        
                        current_score += delta;
                        improvements++;
                        improved = true;
                        vertex_improved = true;
                        
                        char filename[256];
                        snprintf(filename, sizeof(filename), "matching_score_%d.csv", 
                               current_score);
                        write_matching(filename, matching, male_map, female_map);
                        
                        printf("Improvement %d: score = %d (vertex %d with %d) [%ld seconds]\n",
                               improvements, current_score, vertex, target, 
                               time(NULL) - start_time);
                        break;
                    }
                }
            }
            
            if (idx % 1000 == 0) {
                printf("Iteration %d: Processed %d vertices, current score: %d [%ld seconds]\n",
                       iterations, idx, current_score, time(NULL) - start_time);
            }
        }
        
        printf("Completed iteration %d: score = %d, improvements = %d [%ld seconds]\n",
               iterations, current_score, improvements, time(NULL) - start_time);
               
    } while (improved && iterations < 10);
    
    free(vertices);
    
    printf("\nOptimization complete:\n");
    printf("Initial score: %d\n", initial_score);
    printf("Final score: %d\n", current_score);
    printf("Improvement: %.2f%%\n", 
           100.0 * (current_score - initial_score) / initial_score);
    printf("Total improvements: %d\n", improvements);
    printf("Total iterations: %d\n", iterations);
    printf("Total time: %ld seconds\n", time(NULL) - start_time);
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
    
    // Create node maps
    male_map = create_node_map();
    if (!male_map) error_exit("Failed to create male node map");
    
    female_map = create_node_map();
    if (!female_map) {
        free_node_map(male_map);
        error_exit("Failed to create female node map");
    }
    
    // Load graphs
    printf("Loading male graph from %s...\n", argv[1]);
    male_graph = load_graph(argv[1], male_map);
    if (!male_graph) {
        free_node_map(male_map);
        free_node_map(female_map);
        error_exit("Failed to load male graph");
    }
    
    printf("Loading female graph from %s...\n", argv[2]);
    female_graph = load_graph(argv[2], female_map);
    if (!female_graph) {
        free_graph(male_graph);
        free_node_map(male_map);
        free_node_map(female_map);
        error_exit("Failed to load female graph");
    }
    
    // Load initial matching
    printf("Loading initial matching from %s...\n", argv[3]);
    matching = load_matching(argv[3], male_map, female_map);
    if (!matching) {
        free_graph(male_graph);
        free_graph(female_graph);
        free_node_map(male_map);
        free_node_map(female_map);
        error_exit("Failed to load initial matching");
    }
    
    // Optimize matching
    optimize_matching(male_graph, female_graph, matching, male_map, female_map);
    
    // Write final result
    printf("Writing final matching to %s...\n", argv[4]);
    write_matching(argv[4], matching, male_map, female_map);
    
    // Cleanup
    printf("Cleaning up...\n");
    if (matching) free(matching);
    if (male_graph) free_graph(male_graph);
    if (female_graph) free_graph(female_graph);
    if (male_map) free_node_map(male_map);
    if (female_map) free_node_map(female_map);
    
    printf("Done.\n");
    return 0;
}
