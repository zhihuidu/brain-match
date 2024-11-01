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
    double avg_weight;    // Average edge weight
    double max_weight;    // Maximum edge weight
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
void print_edge_stats(Graph *g, int vertex);
void print_matching_progress(int v1, int v2, double struct_score, double align_score, 
                           double total_score, Graph *g1, Graph *g2);
int validate_graphs(Graph *g1, Graph *g2);
double compute_edge_alignment(Graph *g1, Graph *g2, int v1, int v2, int *current_matching, 
                            int *matched, int verbose);
AdjNode* add_edge_to_list(AdjNode* head, int vertex, double weight);
void free_adj_list(AdjNode* head);

// Helper functions
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

int compare_ints(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void print_edge_stats(Graph *g, int vertex) {
    int out_count = 0;
    double out_weight_sum = 0;
    int in_count = 0;
    double in_weight_sum = 0;
    
    for (AdjNode *e = g->out_edges[vertex]; e; e = e->next) {
        out_count++;
        out_weight_sum += e->weight;
    }
    for (AdjNode *e = g->in_edges[vertex]; e; e = e->next) {
        in_count++;
        in_weight_sum += e->weight;
    }
    
    printf("    Edges: %d out (sum=%.1f), %d in (sum=%.1f)\n", 
           out_count, out_weight_sum, in_count, in_weight_sum);
}

void print_matching_progress(int v1, int v2, double struct_score, double align_score, 
                           double total_score, Graph *g1, Graph *g2) {
    printf("\nConsidering match:\n");
    printf("  V1 (ID=%d):\n", g1->node_ids[v1]);
    print_edge_stats(g1, v1);
    printf("  V2 (ID=%d):\n", g2->node_ids[v2]);
    print_edge_stats(g2, v2);
    printf("  Scores: structural=%.2f, alignment=%.2f, total=%.2f\n",
           struct_score, align_score, total_score);
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
    graph->avg_weight = 0.0;
    graph->max_weight = 0.0;

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

int validate_graphs(Graph *g1, Graph *g2) {
    printf("\nValidating graphs...\n");
    
    if (g1->num_nodes != g2->num_nodes) {
        fprintf(stderr, "Error: Graphs have different numbers of nodes (%d vs %d)\n",
                g1->num_nodes, g2->num_nodes);
        return -1;
    }
    
    // Basic sanity checks
    for (int i = 0; i < g1->num_nodes; i++) {
        int g1_edges = 0, g2_edges = 0;
        for (AdjNode *e = g1->out_edges[i]; e; e = e->next) g1_edges++;
        for (AdjNode *e = g1->in_edges[i]; e; e = e->next) g1_edges++;
        for (AdjNode *e = g2->out_edges[i]; e; e = e->next) g2_edges++;
        for (AdjNode *e = g2->in_edges[i]; e; e = e->next) g2_edges++;
        
        if (g1_edges == 0) {
            printf("Warning: Vertex %d in graph 1 is isolated\n", g1->node_ids[i]);
        }
        if (g2_edges == 0) {
            printf("Warning: Vertex %d in graph 2 is isolated\n", g2->node_ids[i]);
        }
    }
    
    // Compute and report graph statistics
    double g1_total_weight = 0, g2_total_weight = 0;
    int g1_total_edges = 0, g2_total_edges = 0;
    
    for (int i = 0; i < g1->num_nodes; i++) {
        for (AdjNode *e = g1->out_edges[i]; e; e = e->next) {
            g1_total_weight += e->weight;
            g1_total_edges++;
            g1->max_weight = fmax(g1->max_weight, e->weight);
        }
        for (AdjNode *e = g2->out_edges[i]; e; e = e->next) {
            g2_total_weight += e->weight;
            g2_total_edges++;
            g2->max_weight = fmax(g2->max_weight, e->weight);
        }
    }
    
    if (g1_total_edges > 0) g1->avg_weight = g1_total_weight / g1_total_edges;
    if (g2_total_edges > 0) g2->avg_weight = g2_total_weight / g2_total_edges;
    
    printf("Graph 1 statistics:\n");
    printf("  Total edges: %d\n", g1_total_edges);
    printf("  Total weight: %.1f\n", g1_total_weight);
    printf("  Average weight: %.2f\n", g1->avg_weight);
    printf("  Maximum weight: %.2f\n", g1->max_weight);
    
    printf("Graph 2 statistics:\n");
    printf("  Total edges: %d\n", g2_total_edges);
    printf("  Total weight: %.1f\n", g2_total_weight);
    printf("  Average weight: %.2f\n", g2->avg_weight);
    printf("  Maximum weight: %.2f\n", g2->max_weight);
    
    return 0;
}

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

double compute_edge_alignment(Graph *g1, Graph *g2, int v1, int v2, int *current_matching, 
                            int *matched, int verbose) {
    double alignment_score = 0.0;
    int edges_considered = 0;
    int edges_matched = 0;
    
    if (verbose) {
        printf("\nComputing edge alignment for vertices %d and %d:\n", 
               g1->node_ids[v1], g2->node_ids[v2]);
    }
    
    // Check outgoing edges
    for (AdjNode *e1 = g1->out_edges[v1]; e1; e1 = e1->next) {
        int target1 = e1->vertex;
        double w1 = e1->weight;
        edges_considered++;
        
        if (matched[target1]) {
            int matched_target = current_matching[target1];
            for (AdjNode *e2 = g2->out_edges[v2]; e2; e2 = e2->next) {
                if (e2->vertex == matched_target) {
                    double contribution = fmin(w1, e2->weight);
                    alignment_score += contribution;
                    edges_matched++;
                    
                    if (verbose) {
                        printf("  Matched edge: %d->%d (w=%.1f) with %d->%d (w=%.1f), contribution=%.1f\n",
                               g1->node_ids[v1], g1->node_ids[target1], w1,
                               g2->node_ids[v2], g2->node_ids[matched_target], e2->weight,
                               contribution);
                    }
                    break;
                }
            }
        }
    }
    
    // Check incoming edges
    for (AdjNode *e1 = g1->in_edges[v1]; e1; e1 = e1->next) {
        int source1 = e1->vertex;
        double w1 = e1->weight;
        edges_considered++;
        
        if (matched[source1]) {
            int matched_source = current_matching[source1];
            for (AdjNode *e2 = g2->in_edges[v2]; e2; e2 = e2->next) {
                if (e2->vertex == matched_source) {
                    double contribution = fmin(w1, e2->weight);
                    alignment_score += contribution;
                    edges_matched++;
                    
                    if (verbose) {
                        printf("  Matched edge: %d<-%d (w=%.1f) with %d<-%d (w=%.1f), contribution=%.1f\n",
                               g1->node_ids[v1], g1->node_ids[source1], w1,
                               g2->node_ids[v2], g2->node_ids[matched_source], e2->weight,
                               contribution);
                    }
                    break;
                }
            }
        }
    }
    
    if (verbose) {
        printf("  Summary: %d/%d edges matched, total alignment score: %.1f\n",
               edges_matched, edges_considered, alignment_score);
    }
    
    return alignment_score;
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
    double total_weight = 0;
    int edge_count = 0;

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

        total_weight += weight;
        edge_count++;
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
    edge_count = 0;
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
    graph->avg_weight = total_weight / edge_count;
    printf("\nAdded %d edges to the graph (avg weight: %.2f)\n", edge_count, graph->avg_weight);
    
    free(temp_nodes);
    fclose(file);
    return graph;
}

void compute_vertex_features(Graph *graph, VertexFeatures *features) {
    if (!graph || !features) return;
    
    int n = graph->num_nodes;
    printf("\nComputing vertex features...\n");
    
    for (int i = 0; i < n; i++) {
        if (i % 1000 == 0) {
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

int compute_feature_distance(VertexFeatures *f1, VertexFeatures *f2) {
    double d1 = fabs((double)f1->in_degree / f1->total_score - 
                     (double)f2->in_degree / f2->total_score);
    double d2 = fabs((double)f1->out_degree / f1->total_score - 
                     (double)f2->out_degree / f2->total_score);
    double d3 = fabs((double)f1->reachable_2hop / f1->total_score - 
                     (double)f2->reachable_2hop / f2->total_score);
    double d4 = fabs((double)f1->can_reach_2hop / f1->total_score - 
                     (double)f2->can_reach_2hop / f2->total_score);
    
    return (int)((d1 + d2 + d3 + d4) * 1000);  // Scale up for integer comparison
}

int match_vertices(Graph *g1, Graph *g2, int *matching) {
    if (!g1 || !g2 || !matching) return -1;
    
    int n = g1->num_nodes;
    printf("\nStarting vertex matching process...\n");
    
    // Validate graphs
    if (validate_graphs(g1, g2) != 0) {
        return -1;
    }
    
    // Allocate arrays
    VertexFeatures *features1 = malloc(n * sizeof(VertexFeatures));
    VertexFeatures *features2 = malloc(n * sizeof(VertexFeatures));
    int *matched = calloc(n, sizeof(int));
    int *temp_matching = malloc(n * sizeof(int));
    
    if (!features1 || !features2 || !matched || !temp_matching) {
        fprintf(stderr, "Memory allocation failed\n");
        free(features1);
        free(features2);
        free(matched);
        free(temp_matching);
        return -1;
    }
    
    // Compute features
    printf("\nComputing features for first graph...\n");
    compute_vertex_features(g1, features1);
    
    printf("\nComputing features for second graph...\n");
    compute_vertex_features(g2, features2);
    
    // Sort vertices by feature scores
    printf("\nSorting vertices by feature scores...\n");
    qsort(features1, n, sizeof(VertexFeatures), compare_features);
    qsort(features2, n, sizeof(VertexFeatures), compare_features);
    
    // Initialize statistics tracking
    double total_alignment = 0.0;
    double best_match_score = -INFINITY;
    double worst_match_score = INFINITY;
    int matches_found = 0;
    
    // Perform matching
    printf("\nPerforming greedy matching with edge alignment...\n");
    
    for (int i = 0; i < n; i++) {
        int v1 = features1[i].original_index;
        double best_score = -INFINITY;
        int best_match = -1;
        double best_struct_score = 0;
        double best_align_score = 0;
        
        // Progress update
        if (i % 100 == 0) {
            printf("\nProgress: %d/%d vertices matched (%.1f%%)\n", 
                   i, n, (100.0 * i) / n);
            printf("Current total alignment: %.1f\n", total_alignment);
        }
        
        // Find best match
        for (int j = 0; j < n; j++) {
            if (!matched[j]) {
                int v2 = features2[j].original_index;
                
                memcpy(temp_matching, matching, n * sizeof(int));
                temp_matching[v1] = g2->node_ids[v2];
                
                double struct_score = -compute_feature_distance(&features1[i], &features2[j]);
                double align_score = compute_edge_alignment(g1, g2, v1, v2, temp_matching, matched, 0);
                double total_score = 0.3 * struct_score + 0.7 * align_score;
                
                if (total_score > best_score) {
                    best_score = total_score;
                    best_match = j;
                    best_struct_score = struct_score;
                    best_align_score = align_score;
                    
                    // Print detailed info for significant improvements
                    if (total_score > 1.5 * best_score && i % 1000 == 0) {
                        print_matching_progress(v1, v2, struct_score, align_score, total_score, g1, g2);
                    }
                }
            }
        }
        
        if (best_match != -1) {
            matching[v1] = g2->node_ids[features2[best_match].original_index];
            matched[best_match] = 1;
            matches_found++;
            total_alignment += best_align_score;
            
            // Update statistics
            best_match_score = fmax(best_match_score, best_score);
            worst_match_score = fmin(worst_match_score, best_score);
            
            // Print significant matches
            if (best_score > 0.8 * best_match_score && i % 1000 == 0) {
                printf("\nFound high-quality match:\n");
                printf("  Vertex %d -> %d (score: %.2f, struct: %.2f, align: %.2f)\n", 
                       g1->node_ids[v1], matching[v1],
                       best_score, best_struct_score, best_align_score);
            }
        }
    }
    
    // Print final statistics
    printf("\nMatching completed:\n");
    printf("  Vertices matched: %d/%d\n", matches_found, n);
    printf("  Total alignment score: %.2f\n", total_alignment);
    printf("  Best match score: %.2f\n", best_match_score);
    printf("  Worst match score: %.2f\n", worst_match_score);
    printf("  Average match score: %.2f\n", total_alignment / matches_found);
    
    // Clean up
    free(temp_matching);
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
