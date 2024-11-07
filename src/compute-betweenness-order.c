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
    int* nodes;
    int* distances;
    double* sigma;      // number of shortest paths
    double* delta;      // dependency values
    int* predecessors;  // array of predecessor lists
    int* pred_counts;   // count of predecessors for each node
    int queue[MAX_NODES];
    int queue_start;
    int queue_end;
} BFSData;

typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap* edges;
    int node_capacity;
} Graph;

// Compare function for sorting nodes by centrality (descending order)
int compare_nodes(const void* a, const void* b) {
    const NodeScore* na = (const NodeScore*)a;
    const NodeScore* nb = (const NodeScore*)b;
    if (nb->centrality > na->centrality) return 1;
    if (nb->centrality < na->centrality) return -1;
    return 0;
}

EdgeMap* new_edge_map() {
    EdgeMap* em = malloc(sizeof(EdgeMap));
    em->capacity = 100;
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    return em;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        em->to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        em->weights = realloc(em->weights, sizeof(int) * em->capacity);
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
    g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
    g->node_capacity = MAX_NODES;
    return g;
}

void add_edge(Graph* g, int from, int to, int weight) {
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    add_to_edge_map(&g->edges[from], to, weight);
}

BFSData* init_bfs_data() {
    BFSData* data = malloc(sizeof(BFSData));
    data->nodes = calloc(NUM_NODES + 1, sizeof(int));
    data->distances = malloc((NUM_NODES + 1) * sizeof(int));
    data->sigma = malloc((NUM_NODES + 1) * sizeof(double));
    data->delta = malloc((NUM_NODES + 1) * sizeof(double));
    data->predecessors = malloc((NUM_NODES + 1) * sizeof(int));
    data->pred_counts = calloc(NUM_NODES + 1, sizeof(int));
    data->queue_start = 0;
    data->queue_end = 0;
    return data;
}

void free_bfs_data(BFSData* data) {
    free(data->nodes);
    free(data->distances);
    free(data->sigma);
    free(data->delta);
    free(data->predecessors);
    free(data->pred_counts);
    free(data);
}

void reset_bfs_data(BFSData* data) {
    memset(data->nodes, 0, (NUM_NODES + 1) * sizeof(int));
    for (int i = 0; i <= NUM_NODES; i++) {
        data->distances[i] = -1;
        data->sigma[i] = 0;
        data->delta[i] = 0;
        data->pred_counts[i] = 0;
    }
    data->queue_start = 0;
    data->queue_end = 0;
}

// Single-source shortest path betweenness calculation
void compute_single_source(Graph* g, int source, BFSData* data, double* centrality) {
    reset_bfs_data(data);
    
    // Initialize for source node
    data->queue[data->queue_end++] = source;
    data->distances[source] = 0;
    data->sigma[source] = 1.0;
    
    // Forward pass - BFS to find shortest paths
    while (data->queue_start != data->queue_end) {
        int v = data->queue[data->queue_start++];
        data->nodes[data->pred_counts[v]++] = v;
        
        for (int i = 0; i < g->edges[v].count; i++) {
            int w = g->edges[v].to_nodes[i];
            
            // First time found this node
            if (data->distances[w] < 0) {
                data->queue[data->queue_end++] = w;
                data->distances[w] = data->distances[v] + 1;
            }
            
            // Found another shortest path
            if (data->distances[w] == data->distances[v] + 1) {
                data->sigma[w] += data->sigma[v];
                data->predecessors[w] = v;
            }
        }
    }
    
    // Backward pass - accumulate dependencies
    while (--data->pred_counts[0] > 0) {
        int w = data->nodes[data->pred_counts[0]];
        if (w == source) continue;
        
        double coeff = (1.0 + data->delta[w]) / data->sigma[w];
        int v = data->predecessors[w];
        data->delta[v] += data->sigma[v] * coeff;
        
        if (w != source) {
            centrality[w] += data->delta[w];
        }
    }
}

void compute_betweenness(Graph* g, double* centrality) {
    BFSData* data = init_bfs_data();
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Sample vertices for approximation
    #pragma omp parallel for
    for (int i = 0; i < NUM_SAMPLES; i++) {
        int source = 1 + rand() % NUM_NODES;
        
        BFSData* local_data = init_bfs_data();
        compute_single_source(g, source, local_data, centrality);
        free_bfs_data(local_data);
        
        if ((i + 1) % 100 == 0) {
            printf("\rProcessed %d/%d samples", i + 1, NUM_SAMPLES);
            fflush(stdout);
        }
    }
    printf("\n");
    
    free_bfs_data(data);
}

Graph* load_graph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    Graph* g = new_graph();
    char line[MAX_LINE_LENGTH];
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    // Read edges
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(g, from, to, weight);
        }
    }
    
    fclose(file);
    return g;
}

void write_ordering(const char* filename, NodeScore* scores) {
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_graph> <output_ordering> <is_reverse>\n", argv[0]);
        return 1;
    }
    
    // Load graph
    printf("Loading graph from %s...\n", argv[1]);
    Graph* g = load_graph(argv[1]);
    
    // Initialize centrality scores
    double* centrality = calloc(NUM_NODES + 1, sizeof(double));
    
    // Compute betweenness centrality
    printf("Computing betweenness centrality...\n");
    compute_betweenness(g, centrality);
    
    // Convert to node scores and sort
    NodeScore* scores = malloc(NUM_NODES * sizeof(NodeScore));
    for (int i = 0; i < NUM_NODES; i++) {
        scores[i].node = i + 1;
        scores[i].centrality = centrality[i + 1];
    }
    
    qsort(scores, NUM_NODES, sizeof(NodeScore), compare_nodes);
    
    // Optionally reverse the ordering
    if (atoi(argv[3])) {
        NodeScore* temp = malloc(NUM_NODES * sizeof(NodeScore));
        for (int i = 0; i < NUM_NODES; i++) {
            temp[i] = scores[NUM_NODES - 1 - i];
        }
        memcpy(scores, temp, NUM_NODES * sizeof(NodeScore));
        free(temp);
    }
    
    // Write results
    printf("Writing ordering to %s...\n", argv[2]);
    write_ordering(argv[2], scores);
    
    // Cleanup
    free(centrality);
    free(scores);
    // Free graph (implementation omitted for brevity)
    
    return 0;
}
