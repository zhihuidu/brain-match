#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1024
#define NUM_NODES 18524
#define NUM_SAMPLES 1000  // Number of random source nodes for approximation
#define MAX_NODES 100000

typedef struct {
    int node;
    double centrality;
} NodeScore;

typedef struct {
    int* predecessors;  // Array of arrays: predecessors[v] = vertices that precede v on shortest paths from s
    int* num_preds;     // Number of predecessors for each vertex  
    int* distances;     // distances[v] = shortest path distance from s to v
    long double* num_paths;  // num_paths[v] = number of shortest paths from s to v
    long double* dependencies; // dependencies[v] = dependency of s on v
    int* queue;         // Queue for BFS
    int* stack;         // Stack recording order vertices are found (for backward pass)
    int queue_head;     // Head of queue for BFS
    int queue_tail;     // Tail of queue for BFS
    int stack_size;     // Current size of stack
} BFSData;


typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap* edges;
    EdgeMap* reverse_edges;
    int node_capacity;
} Graph;

// Compare function for sorting nodes by centrality (descending order)
int compare_nodes(const void* a, const void* b) {
    const NodeScore* na = (const NodeScore*)a;
    const NodeScore* nb = (const NodeScore*)b;
    if (na->centrality > nb->centrality) return -1;
    if (na->centrality < nb->centrality) return 1;
    return 0;
}

EdgeMap* new_edge_map() {
    EdgeMap* em = malloc(sizeof(EdgeMap));
    if (!em) {
        fprintf(stderr, "Failed to allocate EdgeMap\n");
        exit(1);
    }
    em->capacity = 100;
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        fprintf(stderr, "Failed to allocate EdgeMap arrays\n");
        exit(1);
    }
    return em;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            fprintf(stderr, "Failed to reallocate EdgeMap arrays\n");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
    if (!g) {
        fprintf(stderr, "Failed to allocate Graph\n");
        exit(1);
    }
    g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
    g->reverse_edges = calloc(MAX_NODES, sizeof(EdgeMap));
    if (!g->edges || !g->reverse_edges) {
        fprintf(stderr, "Failed to allocate edges arrays\n");
        exit(1);
    }
    g->node_capacity = MAX_NODES;
    return g;
}

void add_edge(Graph* g, int from, int to, int weight) {
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
}

BFSData* init_bfs_data() {
    BFSData* data = malloc(sizeof(BFSData));
    if (!data) {
        fprintf(stderr, "Failed to allocate BFSData\n");
        exit(1);
    }
    
    data->predecessors = malloc(NUM_NODES * NUM_NODES * sizeof(int)); // Worst case all nodes are predecessors
    data->num_preds = calloc(NUM_NODES, sizeof(int));
    data->distances = malloc(NUM_NODES * sizeof(int));
    data->num_paths = malloc(NUM_NODES * sizeof(long double));
    data->dependencies = malloc(NUM_NODES * sizeof(long double));
    data->queue = malloc(NUM_NODES * sizeof(int));
    data->stack = malloc(NUM_NODES * sizeof(int));
    
    if (!data->predecessors || !data->num_preds || !data->distances || 
        !data->num_paths || !data->dependencies || !data->queue || !data->stack) {
        fprintf(stderr, "Failed to allocate BFS arrays\n");
        exit(1);
    }
    
    return data;
}


void free_bfs_data(BFSData* data) {
    if (data) {
        free(data->predecessors);
        free(data->num_preds);
        free(data->distances);
        free(data->num_paths);
        free(data->dependencies);
        free(data->queue);
        free(data->stack);
        free(data);
    }
}

void reset_bfs_data(BFSData* data) {
    // Reset queue and stack pointers
    data->queue_head = 0;
    data->queue_tail = 0;
    data->stack_size = 0;
    
    // Reset arrays
    for (int i = 0; i < NUM_NODES; i++) {
        data->distances[i] = -1;
        data->num_paths[i] = 0;
        data->num_preds[i] = 0;
        data->dependencies[i] = 0;
    }
}

void compute_single_source_shortest_paths(Graph* g, int s, BFSData* data) {
    reset_bfs_data(data);

    // Initialize source
    data->distances[s] = 0;
    data->num_paths[s] = 1;
    data->queue[data->queue_tail++] = s;
    
    // Forward pass - BFS to find shortest paths
    while (data->queue_head < data->queue_tail) {
        int v = data->queue[data->queue_head++];
        data->stack[data->stack_size++] = v;
        
        for (int i = 0; i < g->edges[v].count; i++) {
            int w = g->edges[v].to_nodes[i];
            
            // First time seeing w
            if (data->distances[w] == -1) {
                data->distances[w] = data->distances[v] + 1;
                data->queue[data->queue_tail++] = w;
            }
            
            // Edge is on a shortest path
            if (data->distances[w] == data->distances[v] + 1) {
                data->num_paths[w] += data->num_paths[v];
                data->predecessors[w * NUM_NODES + data->num_preds[w]++] = v;
            }
        }
    }
    
    // Backward pass - accumulate dependencies
    while (data->stack_size > 0) {
        int w = data->stack[--data->stack_size];
        
        for (int i = 0; i < data->num_preds[w]; i++) {
            int v = data->predecessors[w * NUM_NODES + i];
            double coeff = data->num_paths[v] / data->num_paths[w];
            data->dependencies[v] += coeff * (1.0 + data->dependencies[w]);
        }
        
        if (w != s) {
            data->dependencies[w] /= 2; // Count each shortest path only once for undirected graph
        }
    }
}

void compute_betweenness(Graph* g, double* centrality) {
    BFSData* data = init_bfs_data(NUM_NODES);
    
    // Initialize centrality
    for(int i = 0; i < NUM_NODES; i++) {
        centrality[i] = 0;
    }
    
    // For each source vertex
    for(int s = 0; s < NUM_SAMPLES; s++) {
        int source = 1 + (rand() % NUM_NODES);
        
        // Compute shortest paths from s
        compute_single_source_shortest_paths(g, source, data);
        
        // Add dependencies to centrality
        for(int v = 0; v < NUM_NODES; v++) {
            if(v != source) {
                centrality[v] += data->dependencies[v];
            }
        }
        
        if((s+1) % 100 == 0) {
            printf("\rProcessed %d/%d samples", s+1, NUM_SAMPLES);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Normalize
    for(int v = 0; v < NUM_NODES; v++) {
        centrality[v] /= ((NUM_SAMPLES) * (NUM_SAMPLES - 1)); // n * (n-1) for directed
    }
    
    // Print some values to verify
    printf("Sample centrality values:\n");
    for(int i = 0; i < 5; i++) {
        printf("Node %d: %.10f\n", i+1, centrality[i]);
    }
}

Graph* load_graph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    Graph* g = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    printf("Loading graph...\n");
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(g, from, to, weight);
            line_count++;
            if (line_count % 100000 == 0) {
                printf("\rLoaded %d edges", line_count);
                fflush(stdout);
            }
        }
    }
    printf("\nLoaded total of %d edges\n", line_count);
    
    fclose(file);
    return g;
}

void write_ordering(const char* filename, NodeScore* scores) {
    // Print original values
    printf("\nFirst 20 scores before sorting:\n");
    for (int i = 0; i < 20; i++) {
        printf("Node %d: centrality = %.10f\n", scores[i].node, scores[i].centrality);
    }

    printf("\nRange of values:\n");
    double min_val = scores[0].centrality;
    double max_val = scores[0].centrality;
    for (int i = 1; i < NUM_NODES; i++) {
        if (scores[i].centrality < min_val) min_val = scores[i].centrality;
        if (scores[i].centrality > max_val) max_val = scores[i].centrality;
    }
    printf("Min centrality: %.10f\n", min_val);
    printf("Max centrality: %.10f\n", max_val);
    
    // Sort by betweenness value (high to low)
    qsort(scores, NUM_NODES, sizeof(NodeScore), compare_nodes);
    
    // Print sorted values
    printf("\nFirst 20 scores after sorting:\n");
    for (int i = 0; i < 20; i++) {
        printf("Node %d: centrality = %.10f\n", scores[i].node, scores[i].centrality);
    }

    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error creating output file: %s\n", filename);
        exit(1);
    }
    
    fprintf(file, "Node ID,Order\n");
    for (int i = 0; i < NUM_NODES; i++) {
        fprintf(file, "%d,%d\n", scores[i].node, i + 1);
    }
    
    fclose(file);
}

void free_graph(Graph* g) {
    if (g) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        free(g->edges);
        free(g->reverse_edges);
        free(g);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_graph> <output_ordering>\n", argv[0]);
        return 1;
    }
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Load graph
    printf("Loading graph from %s...\n", argv[1]);
    Graph* g = load_graph(argv[1]);
    
    // Initialize centrality scores
    double* centrality = calloc(NUM_NODES + 1, sizeof(double));
    if (!centrality) {
        fprintf(stderr, "Failed to allocate centrality array\n");
        free_graph(g);
        return 1;
    }
    
    // Compute betweenness centrality
    printf("Computing betweenness centrality...\n");
    compute_betweenness(g, centrality);
    
    // Convert to node scores
    NodeScore* scores = malloc(NUM_NODES * sizeof(NodeScore));
    if (!scores) {
        fprintf(stderr, "Failed to allocate scores array\n");
        free(centrality);
        free_graph(g);
        return 1;
    }
    
    for (int i = 0; i < NUM_NODES; i++) {
        scores[i].node = i + 1;
        scores[i].centrality = centrality[i + 1];
    }
    
    // Write results
    printf("Writing ordering to %s...\n", argv[2]);
    write_ordering(argv[2], scores);
    
    // Cleanup
    free_graph(g);
    free(centrality);
    free(scores);
    
    printf("Done!\n");
    return 0;
}
