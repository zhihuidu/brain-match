#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define INITIAL_NODE_CAPACITY 100
#define INITIAL_EDGE_CAPACITY 1000
#define MAX_LINE_LENGTH 256

// Edge structure for sparse representation
typedef struct {
    int src;
    int dst;
    double weight;
} Edge;

// Adjacency list node
typedef struct AdjNode {
    int vertex;
    double weight;
    struct AdjNode* next;
} AdjNode;

// Graph structure using adjacency lists
typedef struct {
    int *node_ids;        // Array to store original node IDs
    int *id_to_index;     // Mapping from original ID to matrix index
    int max_node_id;      // Maximum node ID encountered
    int num_nodes;        // Number of nodes in the graph
    int num_edges;        // Number of edges
    AdjNode **out_edges;  // Array of adjacency lists for outgoing edges
    AdjNode **in_edges;   // Array of adjacency lists for incoming edges
} Graph;

typedef struct {
    int in_degree;
    int out_degree;
    int reachable_2hop;
    int can_reach_2hop;
    int total_score;
    int original_index;
} VertexFeatures;

// Function declarations
Graph* initialize_graph(int num_nodes, int max_node_id);
void free_graph(Graph *graph);
Graph* load_graph(const char *filename, int *num_nodes);
void compute_vertex_features(Graph *graph, VertexFeatures *features);
int compare_features(const void *a, const void *b);
int compare_ints(const void *a, const void *b);
void compute_2hop_metrics(Graph *graph, int vertex_idx, int *reachable, int *can_reach);
int match_vertices(Graph *g1, Graph *g2, int *matching);
int write_matching(const char *filename, int *matching, Graph *g1, Graph *g2);
void print_feature_stats(VertexFeatures *features, int num_nodes, const char *graph_name);
void print_vertex_details(VertexFeatures *features, int num_nodes, const char *graph_name);
AdjNode* add_edge_to_list(AdjNode* head, int vertex, double weight);
void free_adj_list(AdjNode* head);
void get_degrees(Graph *graph, int vertex, int *out_degree, int *in_degree);

// Helper function to get minimum of two integers
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// Compare function for integer sorting
int compare_ints(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void print_feature_stats(VertexFeatures *features, int num_nodes, const char *graph_name) {
    if (!features || num_nodes <= 0) return;

    int min_score = INT_MAX;
    int max_score = INT_MIN;
    long long sum = 0;
    
    for (int i = 0; i < num_nodes; i++) {
        int score = features[i].total_score;
        min_score = (score < min_score) ? score : min_score;
        max_score = (score > max_score) ? score : max_score;
        sum += score;
    }
    
    double avg_score = (double)sum / num_nodes;

    printf("\nFeature Statistics for %s:\n", graph_name);
    printf("  Minimum total score: %d\n", min_score);
    printf("  Maximum total score: %d\n", max_score);
    printf("  Average total score: %.2f\n", avg_score);
    printf("  Number of vertices: %d\n", num_nodes);
}

void print_vertex_details(VertexFeatures *features, int num_nodes, const char *graph_name) {
    printf("\nDetailed vertex features for %s:\n", graph_name);
    printf("NodeIdx\tIn-deg\tOut-deg\t2-hop-out\t2-hop-in\tTotal\n");
    for (int i = 0; i < min(10, num_nodes); i++) {
        printf("%d\t%d\t%d\t%d\t\t%d\t\t%d\n",
               features[i].original_index,
               features[i].in_degree,
               features[i].out_degree,
               features[i].reachable_2hop,
               features[i].can_reach_2hop,
               features[i].total_score);
    }
    if (num_nodes > 10) {
        printf("... and %d more vertices\n", num_nodes - 10);
    }
}

AdjNode* add_edge_to_list(AdjNode* head, int vertex, double weight) {
    AdjNode* new_node = malloc(sizeof(AdjNode));
    if (!new_node) return head;
    
    new_node->vertex = vertex;
    new_node->weight = weight;
    new_node->next = head;
    return new_node;
}

void free_adj_list(AdjNode* head) {
    while (head) {
        AdjNode* temp = head;
        head = head->next;
        free(temp);
    }
}

Graph* initialize_graph(int num_nodes, int max_node_id) {
    if (num_nodes <= 0 || max_node_id < 0) {
        fprintf(stderr, "Invalid parameters for initialize_graph\n");
        return NULL;
    }

    printf("Initializing graph with %d nodes\n", num_nodes);
    Graph *graph = malloc(sizeof(Graph));
    if (!graph) {
        fprintf(stderr, "Memory allocation failed for Graph structure\n");
        return NULL;
    }

    graph->num_nodes = num_nodes;
    graph->max_node_id = max_node_id;
    graph->num_edges = 0;

    // Allocate arrays
    graph->node_ids = malloc(num_nodes * sizeof(int));
    if (!graph->node_ids) {
        fprintf(stderr, "Memory allocation failed for node_ids array\n");
        free(graph);
        return NULL;
    }

    graph->id_to_index = malloc((max_node_id + 1) * sizeof(int));
    if (!graph->id_to_index) {
        fprintf(stderr, "Memory allocation failed for id_to_index array\n");
        free(graph->node_ids);
        free(graph);
        return NULL;
    }

    // Initialize id_to_index with -1
    for (int i = 0; i <= max_node_id; i++) {
        graph->id_to_index[i] = -1;
    }

    // Allocate adjacency lists
    graph->out_edges = calloc(num_nodes, sizeof(AdjNode*));
    if (!graph->out_edges) {
        fprintf(stderr, "Memory allocation failed for out_edges array\n");
        free(graph->id_to_index);
        free(graph->node_ids);
        free(graph);
        return NULL;
    }

    graph->in_edges = calloc(num_nodes, sizeof(AdjNode*));
    if (!graph->in_edges) {
        fprintf(stderr, "Memory allocation failed for in_edges array\n");
        free(graph->out_edges);
        free(graph->id_to_index);
        free(graph->node_ids);
        free(graph);
        return NULL;
    }

    return graph;
}

void free_graph(Graph *graph) {
    if (!graph) return;

    if (graph->out_edges) {
        for (int i = 0; i < graph->num_nodes; i++) {
            free_adj_list(graph->out_edges[i]);
        }
        free(graph->out_edges);
    }

    if (graph->in_edges) {
        for (int i = 0; i < graph->num_nodes; i++) {
            free_adj_list(graph->in_edges[i]);
        }
        free(graph->in_edges);
    }

    if (graph->node_ids) free(graph->node_ids);
    if (graph->id_to_index) free(graph->id_to_index);
    free(graph);
}

// Function to count 2-hop neighbors using adjacency lists
void compute_2hop_metrics(Graph *graph, int vertex_idx, int *reachable, int *can_reach) {
    if (!graph || !reachable || !can_reach) return;

    int n = graph->num_nodes;
    *reachable = 0;
    *can_reach = 0;

    // Allocate visited arrays
    char *visited = calloc(n, sizeof(char));
    if (!visited) {
        fprintf(stderr, "Memory allocation failed for visited array\n");
        return;
    }

    // Count vertices reachable in 2 hops
    for (AdjNode *edge1 = graph->out_edges[vertex_idx]; edge1; edge1 = edge1->next) {
        for (AdjNode *edge2 = graph->out_edges[edge1->vertex]; edge2; edge2 = edge2->next) {
            if (!visited[edge2->vertex]) {
                visited[edge2->vertex] = 1;
                (*reachable)++;
            }
        }
    }

    // Reset visited array
    memset(visited, 0, n * sizeof(char));

    // Count vertices that can reach this one in 2 hops
    for (AdjNode *edge1 = graph->in_edges[vertex_idx]; edge1; edge1 = edge1->next) {
        for (AdjNode *edge2 = graph->in_edges[edge1->vertex]; edge2; edge2 = edge2->next) {
            if (!visited[edge2->vertex]) {
                visited[edge2->vertex] = 1;
                (*can_reach)++;
            }
        }
    }

    free(visited);
}

void compute_vertex_features(Graph *graph, VertexFeatures *features) {
    if (!graph || !features) return;
    
    int n = graph->num_nodes;
    printf("\nComputing vertex features...\n");
    
    for (int i = 0; i < n; i++) {
        if (i % 1000 == 0) {  // Progress update every 1000 vertices
            printf("  Processing vertex %d of %d (%.1f%%)\r", 
                   i, n, (100.0 * i) / n);
            fflush(stdout);
        }
        
        features[i].original_index = i;
        
        // Count degrees using adjacency lists
        int out_degree = 0;
        int in_degree = 0;
        
        // Count out-degree
        for (AdjNode *curr = graph->out_edges[i]; curr; curr = curr->next) {
            out_degree++;
        }
        
        // Count in-degree
        for (AdjNode *curr = graph->in_edges[i]; curr; curr = curr->next) {
            in_degree++;
        }
        
        features[i].out_degree = out_degree;
        features[i].in_degree = in_degree;
        
        // Compute 2-hop metrics
        compute_2hop_metrics(graph, i, &features[i].reachable_2hop, &features[i].can_reach_2hop);
        
        // Compute total score
        features[i].total_score = features[i].in_degree + features[i].out_degree + 
                                features[i].reachable_2hop + features[i].can_reach_2hop;
    }
    printf("\nFeature computation completed for %d vertices\n", n);
}

int compare_features(const void *a, const void *b) {
    const VertexFeatures *fa = (const VertexFeatures *)a;
    const VertexFeatures *fb = (const VertexFeatures *)b;
    return fb->total_score - fa->total_score;  // Sort in descending order
}

Graph* load_graph(const char *filename, int *num_nodes) {
    if (!filename || !num_nodes) return NULL;

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    int capacity = INITIAL_NODE_CAPACITY;
    int *temp_nodes = malloc(capacity * sizeof(int));
    if (!temp_nodes) {
        fprintf(stderr, "Memory allocation failed for temporary nodes array\n");
        fclose(file);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int node_count = 0;
    int max_node_id = -1;

    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Error reading header from file\n");
        free(temp_nodes);
        fclose(file);
        return NULL;
    }

    printf("Scanning file to identify nodes...\n");
    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) continue;

        max_node_id = src > max_node_id ? src : max_node_id;
        max_node_id = tgt > max_node_id ? tgt : max_node_id;

        // Process source node
        int src_found = 0;
        for (int i = 0; i < node_count; i++) {
            if (temp_nodes[i] == src) {
                src_found = 1;
                break;
            }
        }
        if (!src_found) {
            if (node_count >= capacity) {
                capacity *= 2;
                int *new_temp = realloc(temp_nodes, capacity * sizeof(int));
                if (!new_temp) {
                    fprintf(stderr, "Memory reallocation failed\n");
                    free(temp_nodes);
                    fclose(file);
                    return NULL;
                }
                temp_nodes = new_temp;
            }
            temp_nodes[node_count++] = src;
        }

        // Process target node
        int tgt_found = 0;
        for (int i = 0; i < node_count; i++) {
            if (temp_nodes[i] == tgt) {
                tgt_found = 1;
                break;
            }
        }
        if (!tgt_found) {
            if (node_count >= capacity) {
                capacity *= 2;
                int *new_temp = realloc(temp_nodes, capacity * sizeof(int));
                if (!new_temp) {
                    fprintf(stderr, "Memory reallocation failed\n");
                    free(temp_nodes);
                    fclose(file);
                    return NULL;
                }
                temp_nodes = new_temp;
            }
            temp_nodes[node_count++] = tgt;
        }
    }

    printf("Found %d unique nodes\n", node_count);
    *num_nodes = node_count;

    // Sort node IDs for consistent mapping
    qsort(temp_nodes, node_count, sizeof(int), compare_ints);

    // Initialize graph
    Graph *graph = initialize_graph(node_count, max_node_id);
    if (!graph) {
        free(temp_nodes);
        fclose(file);
        return NULL;
    }

    // Create node ID mapping
    memcpy(graph->node_ids, temp_nodes, node_count * sizeof(int));
    for (int i = 0; i < node_count; i++) {
        graph->id_to_index[temp_nodes[i]] = i;
    }

    // Second pass: add edges
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header

    printf("Building adjacency lists...\n");
    int edge_count = 0;
    int progress = 0;
    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) continue;

        int src_idx = graph->id_to_index[src];
        int tgt_idx = graph->id_to_index[tgt];

        if (src_idx != -1 && tgt_idx != -1) {
            // Add to outgoing edges
            graph->out_edges[src_idx] = add_edge_to_list(graph->out_edges[src_idx], tgt_idx, weight);
            // Add to incoming edges
            graph->in_edges[tgt_idx] = add_edge_to_list(graph->in_edges[tgt_idx], src_idx, weight);
            edge_count++;
            
            // Progress reporting
            if (++progress % 1000000 == 0) {
                printf("  Processed %d million edges\r", progress / 1000000);
                fflush(stdout);
            }
        }
    }

    graph->num_edges = edge_count;
    printf("\nAdded %d edges to the graph\n", edge_count);
    free(temp_nodes);
    fclose(file);
    return graph;
}

int compute_feature_distance(VertexFeatures *f1, VertexFeatures *f2) {
    return abs(f1->in_degree - f2->in_degree) +
           abs(f1->out_degree - f2->out_degree) +
           abs(f1->reachable_2hop - f2->reachable_2hop) +
           abs(f1->can_reach_2hop - f2->can_reach_2hop);
}

int match_vertices(Graph *g1, Graph *g2, int *matching) {
    if (!g1 || !g2 || !matching) return -1;
    
    int n = g1->num_nodes;
    printf("\nStarting vertex matching process...\n");
    
    // Allocate and compute features for both graphs
    VertexFeatures *features1 = malloc(n * sizeof(VertexFeatures));
    if (!features1) {
        fprintf(stderr, "Memory allocation failed for features1\n");
        return -1;
    }

    VertexFeatures *features2 = malloc(n * sizeof(VertexFeatures));
    if (!features2) {
        fprintf(stderr, "Memory allocation failed for features2\n");
        free(features1);
        return -1;
    }

    int *matched = calloc(n, sizeof(int));
    if (!matched) {
        fprintf(stderr, "Memory allocation failed for matched array\n");
        free(features1);
        free(features2);
        return -1;
    }
    
    printf("\nComputing features for first graph...\n");
    compute_vertex_features(g1, features1);
    print_feature_stats(features1, n, "Graph 1");
    print_vertex_details(features1, n, "Graph 1");
    
    printf("\nComputing features for second graph...\n");
    compute_vertex_features(g2, features2);
    print_feature_stats(features2, n, "Graph 2");
    print_vertex_details(features2, n, "Graph 2");
    
    printf("\nSorting vertices by feature scores...\n");
    qsort(features1, n, sizeof(VertexFeatures), compare_features);
    qsort(features2, n, sizeof(VertexFeatures), compare_features);
    
    // Perform greedy matching
    printf("\nPerforming greedy matching...\n");
    int matches_found = 0;
    for (int i = 0; i < n; i++) {
        if (i % 1000 == 0) {  // Progress update every 1000 matches
            printf("  Matching vertex %d of %d (%.1f%%)\r", 
                   i, n, (100.0 * i) / n);
            fflush(stdout);
        }

        int min_distance = INT_MAX;
        int best_match = -1;
        
        // Find the best unmatched vertex in g2
        for (int j = 0; j < n; j++) {
            if (!matched[j]) {
                int distance = compute_feature_distance(&features1[i], &features2[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_match = j;
                }
            }
        }
        
        if (best_match != -1) {
            // Store the matching using original node IDs
            matching[features1[i].original_index] = g2->node_ids[features2[best_match].original_index];
            matched[best_match] = 1;
            matches_found++;
        }
    }
    
    printf("\nMatching completed: %d vertices matched\n", matches_found);
    
    free(features1);
    free(features2);
    free(matched);
    return 0;
}

int write_matching(const char *filename, int *matching, Graph *g1, Graph *g2) {
    if (!filename || !matching || !g1 || !g2) return -1;

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return -1;
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < g1->num_nodes; i++) {
        fprintf(file, "%d,%d\n", g1->node_ids[i], matching[i]);
    }

    fclose(file);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("Starting graph matching process...\n");
    printf("Input files:\n");
    printf("  Graph 1: %s\n", argv[1]);
    printf("  Graph 2: %s\n", argv[2]);
    printf("  Output: %s\n", argv[3]);

    // Load first graph
    printf("\nLoading first graph...\n");
    int num_nodes1, num_nodes2;
    Graph *g1 = load_graph(argv[1], &num_nodes1);
    if (!g1) {
        fprintf(stderr, "Failed to load first graph from %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    printf("Successfully loaded %d nodes from first graph\n", num_nodes1);

    // Load second graph
    printf("\nLoading second graph...\n");
    Graph *g2 = load_graph(argv[2], &num_nodes2);
    if (!g2) {
        fprintf(stderr, "Failed to load second graph from %s\n", argv[2]);
        free_graph(g1);
        return EXIT_FAILURE;
    }
    printf("Successfully loaded %d nodes from second graph\n", num_nodes2);

    // Verify graphs have same number of nodes
    if (num_nodes1 != num_nodes2) {
        fprintf(stderr, "Error: Graphs have different numbers of nodes (%d vs %d)\n",
                num_nodes1, num_nodes2);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    // Allocate matching array
    int *matching = malloc(num_nodes1 * sizeof(int));
    if (!matching) {
        fprintf(stderr, "Memory allocation failed for matching array\n");
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    // Perform matching
    printf("\nPerforming vertex matching...\n");
    int match_result = match_vertices(g1, g2, matching);
    if (match_result != 0) {
        fprintf(stderr, "Error during vertex matching\n");
        free(matching);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    // Write results
    printf("\nWriting matching results...\n");
    int write_result = write_matching(argv[3], matching, g1, g2);
    if (write_result != 0) {
        fprintf(stderr, "Error writing matching results to %s\n", argv[3]);
        free(matching);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    // Clean up
    free(matching);
    free_graph(g1);
    free_graph(g2);

    printf("\nMatching completed successfully. Results written to %s\n", argv[3]);
    return EXIT_SUCCESS;
}
