#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <errno.h>

#define MAX_ITER 10000
#define INITIAL_TEMP 100.0
#define COOLING_RATE 0.995
#define MIN_TEMP 0.01
#define LINE_BUFFER_SIZE 1024

// Data structures
typedef struct {
    int target;
    int weight;
} AdjEdge;

typedef struct {
    AdjEdge *edges;
    int degree;
    int capacity;
} AdjacencyList;

typedef struct {
    AdjacencyList *adj;
    int num_nodes;
} Graph;

typedef struct {
    int *node_ids;
    int count;
    int capacity;
} NodeMap;

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
    if (!ptr) {
        handle_error("Memory allocation failed");
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        handle_error("Memory reallocation failed");
    }
    return new_ptr;
}

// NodeMap functions
NodeMap *create_node_map(int initial_capacity) {
    NodeMap *map = safe_malloc(sizeof(NodeMap));
    map->node_ids = safe_malloc(initial_capacity * sizeof(int));
    map->count = 0;
    map->capacity = initial_capacity;
    return map;
}

int add_node(NodeMap *map, int node_id) {
    for (int i = 0; i < map->count; i++) {
        if (map->node_ids[i] == node_id) {
            return i;
        }
    }

    if (map->count >= map->capacity) {
        map->capacity *= 2;
        map->node_ids = safe_realloc(map->node_ids, map->capacity * sizeof(int));
    }

    map->node_ids[map->count] = node_id;
    return map->count++;
}

void free_node_map(NodeMap *map) {
    if (map) {
        if (map->node_ids) free(map->node_ids);
        free(map);
    }
}

// Graph functions
Graph *create_graph(int num_nodes) {
    Graph *graph = safe_malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->adj = safe_malloc(num_nodes * sizeof(AdjacencyList));
    
    for (int i = 0; i < num_nodes; i++) {
        graph->adj[i].edges = safe_malloc(2 * sizeof(AdjEdge));
        graph->adj[i].degree = 0;
        graph->adj[i].capacity = 2;
    }
    return graph;
}

void add_edge(Graph *graph, int src, int tgt, int weight) {
    if (src >= graph->num_nodes || tgt >= graph->num_nodes) {
        handle_error("Invalid node index in add_edge");
    }

    AdjacencyList *adj = &graph->adj[src];
    
    // Check for existing edge
    for (int i = 0; i < adj->degree; i++) {
        if (adj->edges[i].target == tgt) {
            adj->edges[i].weight = weight;  // Update weight if edge exists
            return;
        }
    }

    // Expand capacity if needed
    if (adj->degree >= adj->capacity) {
        adj->capacity *= 2;
        adj->edges = safe_realloc(adj->edges, adj->capacity * sizeof(AdjEdge));
    }

    // Add new edge
    adj->edges[adj->degree++] = (AdjEdge){tgt, weight};
}

void free_graph(Graph *graph) {
    if (graph) {
        if (graph->adj) {
            for (int i = 0; i < graph->num_nodes; i++) {
                if (graph->adj[i].edges) {
                    free(graph->adj[i].edges);
                }
            }
            free(graph->adj);
        }
        free(graph);
    }
}

// File loading functions
Graph *load_graph(const char *filename, NodeMap *map) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        handle_error("Cannot open graph file");
    }

    char line[LINE_BUFFER_SIZE];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        handle_error("Empty graph file");
    }

    Graph *graph = NULL;
    long line_number = 1;

    while (fgets(line, sizeof(line), file)) {
        line_number++;
        int src_id, tgt_id, weight;
        
        line[strcspn(line, "\n")] = 0;
        
        if (sscanf(line, "%d,%d,%d", &src_id, &tgt_id, &weight) != 3) {
            fprintf(stderr, "Warning: Invalid format at line %ld: %s\n", line_number, line);
            continue;
        }

        if (weight < 0) {
            fprintf(stderr, "Warning: Negative weight at line %ld\n", line_number);
        }

        int src = add_node(map, src_id);
        int tgt = add_node(map, tgt_id);

        if (!graph) {
            graph = create_graph(map->capacity);
        } else if (src >= graph->num_nodes || tgt >= graph->num_nodes) {
            int new_size = map->capacity;
            Graph *new_graph = create_graph(new_size);
            
            // Copy existing edges
            for (int i = 0; i < graph->num_nodes; i++) {
                for (int j = 0; j < graph->adj[i].degree; j++) {
                    add_edge(new_graph, i, graph->adj[i].edges[j].target,
                            graph->adj[i].edges[j].weight);
                }
            }
            
            free_graph(graph);
            graph = new_graph;
        }

        add_edge(graph, src, tgt, weight);
    }

    fclose(file);
    
    if (!graph) {
        handle_error("No valid data found in graph file");
    }
    
    return graph;
}

int *load_initial_matching(const char *filename, NodeMap *map_g1, NodeMap *map_g2) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        handle_error("Cannot open initial matching file");
    }

    int *matching = safe_malloc(map_g1->count * sizeof(int));
    for (int i = 0; i < map_g1->count; i++) {
        matching[i] = -1;
    }

    char line[LINE_BUFFER_SIZE];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        free(matching);
        handle_error("Empty matching file");
    }

    long line_number = 1;
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        int node_g1_id, node_g2_id;
        
        line[strcspn(line, "\n")] = 0;
        
        if (sscanf(line, "%d,%d", &node_g1_id, &node_g2_id) != 2) {
            fprintf(stderr, "Warning: Invalid format at line %ld: %s\n", line_number, line);
            continue;
        }

        int mapped_g1 = add_node(map_g1, node_g1_id);
        int mapped_g2 = add_node(map_g2, node_g2_id);

        if (matching[mapped_g1] != -1) {
            fprintf(stderr, "Warning: Node %d already matched at line %ld\n", node_g1_id, line_number);
        }
        
        matching[mapped_g1] = mapped_g2;
    }

    fclose(file);

    // Verify complete matching
    for (int i = 0; i < map_g1->count; i++) {
        if (matching[i] == -1) {
            fprintf(stderr, "Warning: Node %d has no match\n", map_g1->node_ids[i]);
        }
    }

    return matching;
}

// Score calculation functions
int calculate_edge_difference(Graph *g1, Graph *g2, int node1, int match1,
                            int target1, int match_target) {
    int weight1 = 0;
    int weight2 = 0;
    int found1 = 0;
    int found2 = 0;

    // Find weight in g1
    for (int i = 0; i < g1->adj[node1].degree; i++) {
        if (g1->adj[node1].edges[i].target == target1) {
            weight1 = g1->adj[node1].edges[i].weight;
            found1 = 1;
            break;
        }
    }

    // Find weight in g2
    for (int i = 0; i < g2->adj[match1].degree; i++) {
        if (g2->adj[match1].edges[i].target == match_target) {
            weight2 = g2->adj[match1].edges[i].weight;
            found2 = 1;
            break;
        }
    }

    if (found1 && found2) {
        return abs(weight1 - weight2);
    } else if (found1) {
        return weight1;
    } else if (found2) {
        return weight2;
    }
    
    return 0;
}

int calculate_total_score(Graph *g1, Graph *g2, int *matching) {
    int total_score = 0;
    
    for (int i = 0; i < g1->num_nodes; i++) {
        if (matching[i] < 0 || matching[i] >= g2->num_nodes) {
            continue;
        }
        
        for (int j = 0; j < g1->adj[i].degree; j++) {
            int target = g1->adj[i].edges[j].target;
            if (target < 0 || target >= g1->num_nodes || 
                matching[target] < 0 || matching[target] >= g2->num_nodes) {
                continue;
            }
            total_score += calculate_edge_difference(g1, g2, i, matching[i],
                                                  target, matching[target]);
        }
    }
    
    return total_score;
}

int calculate_swap_delta(Graph *g1, Graph *g2, int *matching,
                        int node1, int node2) {
    int before_score = 0;
    int after_score = 0;

    // Calculate score before swap
    for (int i = 0; i < g1->adj[node1].degree; i++) {
        int target = g1->adj[node1].edges[i].target;
        before_score += calculate_edge_difference(g1, g2, node1, matching[node1],
                                               target, matching[target]);
    }
    
    for (int i = 0; i < g1->adj[node2].degree; i++) {
        int target = g1->adj[node2].edges[i].target;
        before_score += calculate_edge_difference(g1, g2, node2, matching[node2],
                                               target, matching[target]);
    }

    // Temporarily swap matches
    int temp = matching[node1];
    matching[node1] = matching[node2];
    matching[node2] = temp;

    // Calculate score after swap
    for (int i = 0; i < g1->adj[node1].degree; i++) {
        int target = g1->adj[node1].edges[i].target;
        after_score += calculate_edge_difference(g1, g2, node1, matching[node1],
                                              target, matching[target]);
    }
    
    for (int i = 0; i < g1->adj[node2].degree; i++) {
        int target = g1->adj[node2].edges[i].target;
        after_score += calculate_edge_difference(g1, g2, node2, matching[node2],
                                              target, matching[target]);
    }

    // Restore original matching
    matching[node2] = matching[node1];
    matching[node1] = temp;

    return after_score - before_score;
}

// Optimization function
void optimize_matching(Graph *g1, Graph *g2, int *matching) {
    if (g1->num_nodes != g2->num_nodes) {
        handle_error("Graphs must have the same number of nodes");
    }

    int num_nodes = g1->num_nodes;
    int current_score = calculate_total_score(g1, g2, matching);
    int best_score = current_score;
    
    // Allocate and copy best matching
    int *best_matching = safe_malloc(num_nodes * sizeof(int));
    memcpy(best_matching, matching, num_nodes * sizeof(int));

    double temperature = INITIAL_TEMP;
    srand(time(NULL));

    for (int iter = 0; iter < MAX_ITER && temperature > MIN_TEMP; iter++) {
        int node1 = rand() % num_nodes;
        int node2 = rand() % num_nodes;
        
        if (node1 == node2) continue;

        int delta = calculate_swap_delta(g1, g2, matching, node1, node2);
        
        if (delta < 0 || (exp(-delta / temperature) > (double)rand() / RAND_MAX)) {
            int temp = matching[node1];
            matching[node1] = matching[node2];
            matching[node2] = temp;
            
            current_score += delta;

            if (current_score < best_score) {
                best_score = current_score;
                memcpy(best_matching, matching, num_nodes * sizeof(int));
            }
        }

        temperature *= COOLING_RATE;
    }

    memcpy(matching, best_matching, num_nodes * sizeof(int));
    free(best_matching);
}

void write_matching(const char *filename, int *matching, int num_nodes,
                   NodeMap *map_g1, NodeMap *map_g2) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        handle_error("Cannot open output file for writing");
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(file, "%d,%d\n", map_g1->node_ids[i], map_g2->node_ids[matching[i]]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <initial_matching.csv> <output_matching.csv>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize pointers to NULL for safe cleanup
    NodeMap *map_g1 = NULL;
    NodeMap *map_g2 = NULL;
    Graph *g1 = NULL;
    Graph *g2 = NULL;
    int *matching = NULL;

    // Create node maps with error checking
    map_g1 = create_node_map(100);
    if (!map_g1) {
        handle_error("Failed to create node map for graph 1");
    }
    
    map_g2 = create_node_map(100);
    if (!map_g2) {
        free_node_map(map_g1);
        handle_error("Failed to create node map for graph 2");
    }

    // Load graphs with error checking
    printf("Loading graph 1...\n");
    g1 = load_graph(argv[1], map_g1);
    if (!g1) {
        free_node_map(map_g1);
        free_node_map(map_g2);
        handle_error("Failed to load graph 1");
    }

    printf("Loading graph 2...\n");
    g2 = load_graph(argv[2], map_g2);
    if (!g2) {
        free_graph(g1);
        free_node_map(map_g1);
        free_node_map(map_g2);
        handle_error("Failed to load graph 2");
    }

    // Verify graph sizes match
    if (map_g1->count != map_g2->count) {
        fprintf(stderr, "Error: Graphs have different numbers of nodes (G1: %d, G2: %d)\n",
                map_g1->count, map_g2->count);
        free_graph(g1);
        free_graph(g2);
        free_node_map(map_g1);
        free_node_map(map_g2);
        return EXIT_FAILURE;
    }

    // Load initial matching with error checking
    printf("Loading initial matching...\n");
    matching = load_initial_matching(argv[3], map_g1, map_g2);
    if (!matching) {
        free_graph(g1);
        free_graph(g2);
        free_node_map(map_g1);
        free_node_map(map_g2);
        handle_error("Failed to load initial matching");
    }

    // Calculate and display initial score
    int initial_score = calculate_total_score(g1, g2, matching);
    printf("Initial matching score: %d\n", initial_score);

    // Optimize matching
    printf("Optimizing matching...\n");
    optimize_matching(g1, g2, matching);

    // Calculate and display final score
    int final_score = calculate_total_score(g1, g2, matching);
    printf("Final matching score: %d\n", final_score);
    
    // Handle improvement percentage calculation
    if (initial_score == 0) {
        if (final_score == 0) {
            printf("No change in score (both initial and final scores are 0)\n");
        } else {
            printf("Score changed from 0 to %d\n", final_score);
        }
    } else {
        double improvement = 100.0 * (initial_score - final_score) / initial_score;
        printf("Improvement: %.2f%%\n", improvement);
    }

    // Write results
    printf("Writing results to %s...\n", argv[4]);
    write_matching(argv[4], matching, map_g1->count, map_g1, map_g2);

    // Clean up - note the order of cleanup
    printf("Cleaning up...\n");
    if (matching) free(matching);
    if (g1) free_graph(g1);
    if (g2) free_graph(g2);
    if (map_g1) free_node_map(map_g1);
    if (map_g2) free_node_map(map_g2);

    printf("Done.\n");
    return EXIT_SUCCESS;
}
