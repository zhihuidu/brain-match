#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define SAVE_INTERVAL 25
#define UPDATE_INTERVAL 20
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define SAVE_INTERVAL 25
#define UPDATE_INTERVAL 20
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MAX_NODES 100000  // Increased from 10000

// Structure to represent a graph node's edges
typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

// Structure to represent the graph
typedef struct Graph {
    EdgeMap* edges;
    EdgeMap* reverse_edges;
    int* nodes;
    int node_count;
    int node_capacity;
} Graph;

// Structure to store node metrics
typedef struct NodeMetrics {
    int in_degree;
    int out_degree;
    int total_weight;
    double avg_in_weight;
    double avg_out_weight;
    int ordering_rank;
} NodeMetrics;

// Function to initialize a new edge map
EdgeMap* new_edge_map() {
    EdgeMap* em = malloc(sizeof(EdgeMap));
    if (!em) {
        printf("Failed to allocate EdgeMap\n");
        exit(1);
    }
    em->capacity = 100;  // Increased initial capacity
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        printf("Failed to allocate EdgeMap arrays\n");
        exit(1);
    }
    return em;
}

// Function to create a new graph
Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
    if (!g) {
        printf("Failed to allocate Graph\n");
        exit(1);
    }
    g->edges = NULL;
    g->reverse_edges = NULL;
    g->nodes = malloc(sizeof(int) * 10000);  // Initial capacity
    if (!g->nodes) {
        printf("Failed to allocate nodes array\n");
        exit(1);
    }
    g->node_count = 0;
    g->node_capacity = 10000;
    return g;
}

// Function to add an edge to the edge map
void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            printf("Failed to reallocate EdgeMap arrays\n");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

// Function to add a node to the graph
void add_node(Graph* g, int node) {
    // Check if node already exists
    for (int i = 0; i < g->node_count; i++) {
        if (g->nodes[i] == node) return;
    }
    
    if (g->node_count >= g->node_capacity) {
        g->node_capacity *= 2;
        int* new_nodes = realloc(g->nodes, sizeof(int) * g->node_capacity);
        if (!new_nodes) {
            printf("Failed to reallocate nodes array\n");
            exit(1);
        }
        g->nodes = new_nodes;
    }
    g->nodes[g->node_count++] = node;
}

// Function to add an edge to the graph
void add_edge(Graph* g, int from, int to, int weight) {
  //    printf("Adding edge: %d -> %d (weight: %d)\n", from, to, weight);
    
    // Initialize edges array if needed
    if (g->edges == NULL) {
        g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
        g->reverse_edges = calloc(MAX_NODES, sizeof(EdgeMap));
        if (!g->edges || !g->reverse_edges) {
            printf("Failed to allocate edges arrays\n");
            exit(1);
        }
    }
    
    // Initialize edge maps if needed
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
    add_node(g, from);
    add_node(g, to);
}

// Function to get edge weight
int get_weight(Graph* g, int from, int to) {
    for (int i = 0; i < g->edges[from].count; i++) {
        if (g->edges[from].to_nodes[i] == to) {
            return g->edges[from].weights[i];
        }
    }
    return 0;
}

// Function to read ordering from CSV
int* read_ordering(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    int* ordering = calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int node_id, order;
        sscanf(line, "%d,%d", &node_id, &order);
        ordering[node_id] = order;
    }
    
    fclose(file);
    return ordering;
}

// Function to calculate node metrics
NodeMetrics* calculate_node_metrics(Graph* g, const char* ordering_path) {
    int max_node = 0;
    for (int i = 0; i < g->node_count; i++) {
        if (g->nodes[i] > max_node) max_node = g->nodes[i];
    }
    
    int* ordering = read_ordering(ordering_path, max_node);
    NodeMetrics* metrics = calloc(max_node + 1, sizeof(NodeMetrics));
    
    for (int i = 0; i < g->node_count; i++) {
        int node = g->nodes[i];
        NodeMetrics* m = &metrics[node];
        
        // Calculate outgoing edges
        m->out_degree = g->edges[node].count;
        for (int j = 0; j < g->edges[node].count; j++) {
            m->total_weight += g->edges[node].weights[j];
        }
        
        // Calculate incoming edges
        m->in_degree = g->reverse_edges[node].count;
        for (int j = 0; j < g->reverse_edges[node].count; j++) {
            m->total_weight += g->reverse_edges[node].weights[j];
        }
        
        if (m->out_degree > 0) {
            m->avg_out_weight = (double)m->total_weight / m->out_degree;
        }
        if (m->in_degree > 0) {
            m->avg_in_weight = (double)m->total_weight / m->in_degree;
        }
        
        m->ordering_rank = ordering[node];
    }
    
    free(ordering);
    return metrics;
}

// Function to calculate node similarity
double calculate_node_similarity(NodeMetrics m1, NodeMetrics m2) {
    double score = 5 * fabs(m1.in_degree - m2.in_degree) +
                   5 * fabs(m1.out_degree - m2.out_degree) +
                   fabs(m1.avg_in_weight - m2.avg_in_weight) +
                   fabs(m1.avg_out_weight - m2.avg_out_weight);
    
    double ordering_sim = fabs(m1.ordering_rank - m2.ordering_rank);
    score = 0.7 * score + 0.3 * ordering_sim;
    
    return -score;
}

// Function to calculate alignment score
int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int score = 0;
    
    for (int i = 0; i < gm->node_count; i++) {
        int src_m = gm->nodes[i];
        for (int j = 0; j < gm->edges[src_m].count; j++) {
            int dst_m = gm->edges[src_m].to_nodes[j];
            int weight_m = gm->edges[src_m].weights[j];
            int src_f = mapping[src_m];
            int dst_f = mapping[dst_m];
            score += MIN(weight_m, get_weight(gf, src_f, dst_f));
        }
    }
    
    return score;
}

// Function to calculate swap delta
int calculate_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2) {
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int delta = 0;
    
    // Handle outgoing edges from node_m1
    for (int i = 0; i < gm->edges[node_m1].count; i++) {
        int node = gm->edges[node_m1].to_nodes[i];
        if (node == node_m2) continue;
        
        int weight_m = gm->edges[node_m1].weights[i];
        int old_f = MIN(weight_m, get_weight(gf, node_f1, mapping[node]));
        int new_f = MIN(weight_m, get_weight(gf, node_f2, mapping[node]));
        delta += new_f - old_f;
    }
    
    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int node = gm->edges[node_m2].to_nodes[i];
        if (node == node_m1) continue;
        
        int weight_m = gm->edges[node_m2].weights[i];
        int old_f = MIN(weight_m, get_weight(gf, node_f2, mapping[node]));
        int new_f = MIN(weight_m, get_weight(gf, node_f1, mapping[node]));
        delta += new_f - old_f;
    }
    
    // Handle incoming edges to node_m1 and node_m2
    // Similar logic for reverse edges...
    
    // Handle direct edges between node_m1 and node_m2
    int weight_m12 = get_weight(gm, node_m1, node_m2);
    int weight_m21 = get_weight(gm, node_m2, node_m1);
    
    if (weight_m12 > 0) {
        int old_f = MIN(weight_m12, get_weight(gf, node_f1, node_f2));
        int new_f = MIN(weight_m12, get_weight(gf, node_f2, node_f1));
        delta += new_f - old_f;
    }
    if (weight_m21 > 0) {
        int old_f = MIN(weight_m21, get_weight(gf, node_f2, node_f1));
        int new_f = MIN(weight_m21, get_weight(gf, node_f1, node_f2));
        delta += new_f - old_f;
    }
    
    return delta;
}

// Function to write mapping to CSV
void write_mapping(const char* filename, int* mapping, int max_node) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error creating file: %s\n", filename);
        exit(1);
    }
    
    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 0; i <= max_node; i++) {
        if (mapping[i] != 0) { // Assuming 0 means no mapping
            fprintf(file, "m%d,f%d\n", i, mapping[i]);
        }
    }
    
    fclose(file);
}

// Function to optimize mapping
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping, 
                     const char* male_ordering_path, const char* female_ordering_path,
                     const char* out_path) {
    int max_node = 0;
    for (int i = 0; i < gm->node_count; i++) {
        if (gm->nodes[i] > max_node) max_node = gm->nodes[i];
    }
    
    int* current_mapping = malloc(sizeof(int) * (max_node + 1));
    memcpy(current_mapping, initial_mapping, sizeof(int) * (max_node + 1));
    
    NodeMetrics* metrics_m = calculate_node_metrics(gm, male_ordering_path);
    NodeMetrics* metrics_f = calculate_node_metrics(gf, female_ordering_path);
    
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    int outer_node_count = 0;
    int sim_predicted = 0;
    int sim_correct = 0;
    
    // Optimization loop
    for (int i = 0; i < gm->node_count; i++) {
        int node_m1 = gm->nodes[i];
        outer_node_count++;
        
        if (outer_node_count % UPDATE_INTERVAL == 0) {
            printf("%d nodes processed of %d\n", outer_node_count, gm->node_count);
            printf("Predicted: %d, Correct: %d, Accuracy: %.2f\n", 
                   sim_predicted, sim_correct, 
                   (double)sim_correct / sim_predicted);
            printf("Current score: %d\n", current_score);
        }
        
        if (outer_node_count % SAVE_INTERVAL == 0) {
            write_mapping(out_path, current_mapping, max_node);
        }
        
        for (int j = 0; j < gm->node_count; j++) {
            int node_m2 = gm->nodes[j];
            if (node_m1 == node_m2) continue;
            
            int node_f1 = current_mapping[node_m1];
            int node_f2 = current_mapping[node_m2];
            
            double current_sim = calculate_node_similarity(metrics_m[node_m1], metrics_f[node_f1]) +
                                calculate_node_similarity(metrics_m[node_m2], metrics_f[node_f2]);
            
            double swapped_sim = calculate_node_similarity(metrics_m[node_m1], metrics_f[node_f2]) +
                                calculate_node_similarity(metrics_m[node_m2], metrics_f[node_f1]);
            
            if (swapped_sim > current_sim) {
                sim_predicted++;
                int delta = calculate_swap_delta(gm, gf, current_mapping, node_m1, node_m2);
                
                if (delta > 0) {
                    sim_correct++;
                    // Swap the mappings
                    current_mapping[node_m1] = node_f2;
                    current_mapping[node_m2] = node_f1;
                    current_score += delta;
                }
            }
        }
    }
    
    free(metrics_m);
    free(metrics_f);
    return current_mapping;
}

// Function to load graph from CSV
Graph* load_graph_from_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    
    Graph* graph = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(graph, from, to, weight);
            line_count++;
            if (line_count % 1000 == 0) {
                printf("Processed %d lines\n", line_count);
            }
        } else {
            printf("Warning: Malformed line in CSV: %s", line);
        }
    }
    
    fclose(file);
    printf("Finished loading graph with %d nodes\n", graph->node_count);
    return graph;
}

// Function to load benchmark mapping from CSV
int* load_benchmark_mapping(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    
    int* mapping = calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char male_str[20], female_str[20];
        sscanf(line, "%[^,],%s", male_str, female_str);
        
        // Remove 'm' and 'f' prefixes and convert to integers
        int male_id = atoi(male_str + 1);  // Skip 'm'
        int female_id = atoi(female_str + 1);  // Skip 'f'
        
        mapping[male_id] = female_id;
    }
    
    fclose(file);
    return mapping;
}

// Function to get maximum node ID from graph
int get_max_node(Graph* g) {
    int max_node = 0;
    for (int i = 0; i < g->node_count; i++) {
        if (g->nodes[i] > max_node) {
            max_node = g->nodes[i];
        }
    }
    return max_node;
}

// Function to clean up graph memory
void free_graph(Graph* g) {
    if (g->edges) {
        for (int i = 0; i < 10000; i++) {  // Using same max as in new_graph
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
        }
        free(g->edges);
    }
    
    if (g->reverse_edges) {
        for (int i = 0; i < 10000; i++) {
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        free(g->reverse_edges);
    }
    
    free(g->nodes);
    free(g);
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        printf("Usage: %s <male graph> <female graph> <male ordering> <female ordering> <in mapping> <out mapping>\n", argv[0]);
        return 1;
    }
    
    printf("Loading male graph from %s\n", argv[1]);
    Graph* gm = load_graph_from_csv(argv[1]);
    if (!gm) {
        printf("Error loading male graph\n");
        return 1;
    }
    printf("Loaded male graph successfully\n");
    
    // Load female graph
    printf("Loading female graph from %s\n", argv[2]);
    Graph* gf = load_graph_from_csv(argv[2]);
    if (!gf) {
        printf("Error loading female graph\n");
        free_graph(gm);
        return 1;
    }
    printf("Loaded in female graph\n");
    
    // Find maximum node ID across both graphs
    int max_node = MAX(get_max_node(gm), get_max_node(gf));
    
    // Load benchmark mapping
    int* benchmark = load_benchmark_mapping(argv[5], max_node);
    if (!benchmark) {
        printf("Error loading benchmark mapping\n");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }
    printf("Loaded in benchmark mapping\n");
    
    // Calculate initial score
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    printf("Initial alignment score: %d\n", initial_score);
    
    printf("Starting optimization...\n");
    
    // Optimize mapping
    int* optimized_mapping = optimize_mapping(gm, gf, benchmark, argv[3], argv[4], argv[6]);
    int optimized_score = calculate_alignment_score(gm, gf, optimized_mapping);
    
    printf("Optimized alignment score: %d\n", optimized_score);
    printf("Improvement: %.2f%%\n", 
           (double)(optimized_score - initial_score) / initial_score * 100.0);
    
    // Write final mapping to CSV
    write_mapping(argv[6], optimized_mapping, max_node);
    
    // Clean up
    free_graph(gm);
    free_graph(gf);
    free(benchmark);
    free(optimized_mapping);
    
    return 0;
}
