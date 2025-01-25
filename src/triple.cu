#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
//#include <omp.h>

#include <cstdlib>  // for malloc
#include <cstdio>  // for io



// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

__global__ void calculateSwapDeltasKernel(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    int* deltas,
    int* node_pairs
);

#endif

// cuda_kernels.cu
//#include "cuda_kernels.cuh"



// Add these constants at the top
#define BATCH_SIZE  270000  // Adjust based on GPU memory
#define NUM_PERMUTATIONS 6  // 3! possible permutations


__global__ void calculateThreeNodeSwapKernel(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    long int batch_start,
    int batch_size,
    int* deltas,
    int* node_triplets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Convert global index to triplet index
    long int global_idx = batch_start + idx;
    int n = num_nodes;
    int matrix_dim = num_nodes + 1;
    
    // Convert linear index to three nodes
    int node_m1 = 1;
    long int temp1 = (long int) (n-1) * (n-2) / 6;
    while (global_idx >= temp1) {
        global_idx -= temp1;
        node_m1++;
        n--;
        temp1 = (long int) (n-1) * (n-2) / 6;
    }
    
    int node_m2 = node_m1 + 1;
    temp1 = n - 2;
    while (global_idx >= temp1) {
        global_idx -= temp1;
        node_m2++;
        temp1--;
    }
    
    int node_m3 = node_m2 + global_idx + 1;
    
    // Get current mappings
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int node_f3 = mapping[node_m3];
    
    // Try all 6 possible permutations and find the best one
    int permutations[6][3] = {
        {node_f1, node_f2, node_f3},  // Original
        {node_f1, node_f3, node_f2},  // Swap 2,3
        {node_f2, node_f1, node_f3},  // Swap 1,2
        {node_f2, node_f3, node_f1},  // 231
        {node_f3, node_f1, node_f2},  // 312
        {node_f3, node_f2, node_f1}   // 321
    };
    
    int best_delta = 0;
    int best_perm = 0;
    
    // Calculate delta for each permutation
    for (int p = 1; p < 6; p++) {  // Start from 1 as 0 is original mapping
        int delta = 0;
        int new_f1 = permutations[p][0];
        int new_f2 = permutations[p][1];
        int new_f3 = permutations[p][2];
        
        // Calculate delta for node_m1 connections to rest of graph
        for (int i = 1; i <= num_nodes; i++) {
            if (i == node_m1 || i == node_m2 || i == node_m3) continue;
            
            // Outgoing edges from node_m1
            if (adj_matrix_m[node_m1 * matrix_dim + i] > 0) {
                int weight = adj_matrix_m[node_m1 * matrix_dim + i];
                int dst_f = mapping[i];
                delta += min(weight, adj_matrix_f[new_f1 * matrix_dim + dst_f]) -
                        min(weight, adj_matrix_f[node_f1 * matrix_dim + dst_f]);
            }
            
            // Incoming edges to node_m1
            if (adj_matrix_m[i * matrix_dim + node_m1] > 0) {
                int weight = adj_matrix_m[i * matrix_dim + node_m1];
                int src_f = mapping[i];
                delta += min(weight, adj_matrix_f[src_f * matrix_dim + new_f1]) -
                        min(weight, adj_matrix_f[src_f * matrix_dim + node_f1]);
            }
        }
        
        // Calculate delta for node_m2 connections to rest of graph
        for (int i = 1; i <= num_nodes; i++) {
            if (i == node_m1 || i == node_m2 || i == node_m3) continue;
            
            // Outgoing edges from node_m2
            if (adj_matrix_m[node_m2 * matrix_dim + i] > 0) {
                int weight = adj_matrix_m[node_m2 * matrix_dim + i];
                int dst_f = mapping[i];
                delta += min(weight, adj_matrix_f[new_f2 * matrix_dim + dst_f]) -
                        min(weight, adj_matrix_f[node_f2 * matrix_dim + dst_f]);
            }
            
            // Incoming edges to node_m2
            if (adj_matrix_m[i * matrix_dim + node_m2] > 0) {
                int weight = adj_matrix_m[i * matrix_dim + node_m2];
                int src_f = mapping[i];
                delta += min(weight, adj_matrix_f[src_f * matrix_dim + new_f2]) -
                        min(weight, adj_matrix_f[src_f * matrix_dim + node_f2]);
            }
        }
        
        // Calculate delta for node_m3 connections to rest of graph
        for (int i = 1; i <= num_nodes; i++) {
            if (i == node_m1 || i == node_m2 || i == node_m3) continue;
            
            // Outgoing edges from node_m3
            if (adj_matrix_m[node_m3 * matrix_dim + i] > 0) {
                int weight = adj_matrix_m[node_m3 * matrix_dim + i];
                int dst_f = mapping[i];
                delta += min(weight, adj_matrix_f[new_f3 * matrix_dim + dst_f]) -
                        min(weight, adj_matrix_f[node_f3 * matrix_dim + dst_f]);
            }
            
            // Incoming edges to node_m3
            if (adj_matrix_m[i * matrix_dim + node_m3] > 0) {
                int weight = adj_matrix_m[i * matrix_dim + node_m3];
                int src_f = mapping[i];
                delta += min(weight, adj_matrix_f[src_f * matrix_dim + new_f3]) -
                        min(weight, adj_matrix_f[src_f * matrix_dim + node_f3]);
            }
        }
        
        // Calculate deltas between the three nodes themselves
        // m1 -> m2
        int w12 = adj_matrix_m[node_m1 * matrix_dim + node_m2];
        if (w12 > 0) {
            delta += min(w12, adj_matrix_f[new_f1 * matrix_dim + new_f2]) -
                    min(w12, adj_matrix_f[node_f1 * matrix_dim + node_f2]);
        }
        
        // m2 -> m1
        int w21 = adj_matrix_m[node_m2 * matrix_dim + node_m1];
        if (w21 > 0) {
            delta += min(w21, adj_matrix_f[new_f2 * matrix_dim + new_f1]) -
                    min(w21, adj_matrix_f[node_f2 * matrix_dim + node_f1]);
        }
        
        // m2 -> m3
        int w23 = adj_matrix_m[node_m2 * matrix_dim + node_m3];
        if (w23 > 0) {
            delta += min(w23, adj_matrix_f[new_f2 * matrix_dim + new_f3]) -
                    min(w23, adj_matrix_f[node_f2 * matrix_dim + node_f3]);
        }
        
        // m3 -> m2
        int w32 = adj_matrix_m[node_m3 * matrix_dim + node_m2];
        if (w32 > 0) {
            delta += min(w32, adj_matrix_f[new_f3 * matrix_dim + new_f2]) -
                    min(w32, adj_matrix_f[node_f3 * matrix_dim + node_f2]);
        }
        
        // m3 -> m1
        int w31 = adj_matrix_m[node_m3 * matrix_dim + node_m1];
        if (w31 > 0) {
            delta += min(w31, adj_matrix_f[new_f3 * matrix_dim + new_f1]) -
                    min(w31, adj_matrix_f[node_f3 * matrix_dim + node_f1]);
        }
        
        // m1 -> m3
        int w13 = adj_matrix_m[node_m1 * matrix_dim + node_m3];
        if (w13 > 0) {
            delta += min(w13, adj_matrix_f[new_f1 * matrix_dim + new_f3]) -
                    min(w13, adj_matrix_f[node_f1 * matrix_dim + node_f3]);
        }
        
        if (delta > best_delta) {
            best_delta = delta;
            best_perm = p;
        }
    }
    
    deltas[idx] = best_delta;
    node_triplets[idx * 4] = node_m1;
    node_triplets[idx * 4 + 1] = node_m2;
    node_triplets[idx * 4 + 2] = node_m3;
    node_triplets[idx * 4 + 3] = best_perm;
}



__global__ void calculateSwapDeltasKernel(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    int* deltas,
    int* node_pairs
) {




    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = (num_nodes * (num_nodes - 1)) / 2;

    if (idx >= num_pairs) return;

    // Convert linear index to vertex pair
    int node_m1 = 1;
    int temp = num_nodes - 1;
    while (idx >= temp) {
        idx -= temp;
        node_m1++;
        temp--;
    }
    int node_m2 = node_m1 + idx + 1;

    int matrix_dim = num_nodes + 1;  // Add this line
    int delta = 0;
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];



    // Update all matrix indexing to use matrix_dim
    for (int i = 1; i <= num_nodes; i++) {
        if (i == node_m1 || i == node_m2) continue;

        // Outgoing edges from node_m1
        if (adj_matrix_m[node_m1 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m1 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]) -
                    min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]);




        }
        // Update other matrix accesses similarly...


        // Incoming edges to node_m1
        if (adj_matrix_m[i * matrix_dim + node_m1] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m1];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]);
        }
        
        // Outgoing edges from node_m2
        if (adj_matrix_m[node_m2 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m2 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]) - 
                    min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]);
        }
        
        // Incoming edges to node_m2
        if (adj_matrix_m[i * matrix_dim + node_m2] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m2];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]);
        }
    }
    
    // Handle direct edges between swapped nodes
    int m1_to_m2 = adj_matrix_m[node_m1 * matrix_dim + node_m2];
    if (m1_to_m2 > 0) {
        delta += min(m1_to_m2, adj_matrix_f[node_f2 * matrix_dim + node_f1]) - 
                min(m1_to_m2, adj_matrix_f[node_f1 * matrix_dim + node_f2]);
    }
    
    int m2_to_m1 = adj_matrix_m[node_m2 * matrix_dim + node_m1];
    if (m2_to_m1 > 0) {
        delta += min(m2_to_m1, adj_matrix_f[node_f1 * matrix_dim + node_f2]) - 
                min(m2_to_m1, adj_matrix_f[node_f2 * matrix_dim + node_f1]);
    }
    
    deltas[idx] = delta;
    node_pairs[idx * 2] = node_m1;
    node_pairs[idx * 2 + 1] = node_m2;
}




// Add these constants
#define SYNC_INTERVAL 1800  // Sync every 1 hour
#define TAG_SCORE 1
#define TAG_MAPPING 2
#define TAG_TERMINATE 3

#define SAVE_INTERVAL 4200
#define UPDATE_INTERVAL 4000
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MAX_NODES 100000

const int NUM_NODES = 18524;

typedef enum {
    LOG_LEVEL_INFO = 0,
    LOG_LEVEL_DEBUG = 1,
    LOG_LEVEL_ERROR = 2
} LogLevel;

#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO

#define LOG_ERROR(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_ERROR) { \
        fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
    }

#define LOG_INFO(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_INFO) { \
        printf("[INFO] " fmt "\n", ##__VA_ARGS__); \
    }

#define LOG_DEBUG(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_DEBUG) { \
        printf("[DEBUG] %s: " fmt "\n", __func__, ##__VA_ARGS__); \
    }







// Structure definitions
typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap* edges;
    EdgeMap* reverse_edges;
    short** adj_matrix;
    int node_capacity;
} Graph;

typedef struct NodeMetrics {
    int in_degree;
    int out_degree;
    int total_weight;
    double avg_in_weight;
    double avg_out_weight;
    int ordering_rank;
} NodeMetrics;


// Progress bar function
void print_progress(int current, int total, const char* prefix) {
    const int bar_width = 50;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);
    
    printf("\r%s [", prefix);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
    if (current == total) printf("\n");
}

// Function to format numbers with commas
char* format_number(int num) {
    static char formatted[32];
    char temp[32];
    int i = 0, j = 0;
    
    sprintf(temp, "%d", num);
    int len = strlen(temp);
    
    while (len > 0) {
        if (i > 0 && i % 3 == 0) formatted[j++] = ',';
        formatted[j++] = temp[len - 1];
        len--;
        i++;
    }
    formatted[j] = '\0';
    
    // Reverse the string
    for (i = 0; i < j/2; i++) {
        char t = formatted[i];
        formatted[i] = formatted[j-1-i];
        formatted[j-1-i] = t;
    }
    
    return formatted;
}

EdgeMap* new_edge_map() {
    EdgeMap* em = (EdgeMap*)malloc(sizeof(EdgeMap));
    if (!em) {
        LOG_ERROR("Failed to allocate EdgeMap");
        exit(1);
    }
    em->capacity = 100;
    em->count = 0;
    em->to_nodes =(int *) malloc(sizeof(int) * em->capacity);
    em->weights = (int *) malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        LOG_ERROR("Failed to allocate EdgeMap arrays");
        exit(1);
    }
    return em;
}

Graph* new_graph() {
    Graph* g =(Graph *) malloc(sizeof(Graph));
    if (!g) {
        LOG_ERROR("Failed to allocate Graph");
        exit(1);
    }
    g->edges = NULL;
    g->reverse_edges = NULL;
    g->adj_matrix = NULL;
    g->adj_matrix = (short**)malloc((NUM_NODES+1) * sizeof(short*));
    for (int i = 0; i <= NUM_NODES; ++i) {
        g->adj_matrix[i] = (short*)calloc((NUM_NODES+1), sizeof(short)); // Initialize to 0
    }

    //g->nodes = malloc(sizeof(int) * 10000);
    //if (!g->nodes) {
    //    LOG_ERROR("Failed to allocate nodes array");
    //    exit(1);
    //}
    //g->node_count = 0;
    g->node_capacity = 10000;
    return g;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = (int *) realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = (int *) realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            LOG_ERROR("Failed to reallocate EdgeMap arrays");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

//void add_node(Graph* g, int node) {
//    for (int i = 0; i < g->node_count; i++) {
//        if (g->nodes[i] == node) return;
//    }
//    
//    if (g->node_count >= g->node_capacity) {
//        g->node_capacity *= 2;
//        int* new_nodes = realloc(g->nodes, sizeof(int) * g->node_capacity);
//        if (!new_nodes) {
//            LOG_ERROR("Failed to reallocate nodes array");
//            exit(1);
//        }
//        g->nodes = new_nodes;
//    }
//    g->nodes[g->node_count++] = node;
//}

void add_edge(Graph* g, int from, int to, int weight) {
  //    LOG_DEBUG("Adding edge: %d -> %d (weight: %d)", from, to, weight);
    g->adj_matrix[from][to] = weight;
    if (g->edges == NULL) {
        g->edges = (EdgeMap*) calloc(MAX_NODES, sizeof(EdgeMap));
        g->reverse_edges = (EdgeMap*)calloc(MAX_NODES, sizeof(EdgeMap));
        if (!g->edges || !g->reverse_edges) {
            LOG_ERROR("Failed to allocate edges arrays");
            exit(1);
        }
    }
    
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
}

int get_weight(Graph* g, int from, int to) {
    return g->adj_matrix[from][to];
    //for (int i = 0; i < g->edges[from].count; i++) {
    //    if (g->edges[from].to_nodes[i] == to) {
    //        return g->edges[from].weights[i];
    //    }
    //}
    //return 0;
}

int* read_ordering(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Error opening file: %s", filename);
        exit(1);
    }

    int* ordering = (int *) calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    
    LOG_INFO("Reading ordering from %s", filename);
    
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

NodeMetrics* calculate_node_metrics(Graph* g, const char* ordering_path) {
    int max_node = NUM_NODES;
    //for (int i = 0; i < g->node_count; i++) {
    //    if (g->nodes[i] > max_node) max_node = g->nodes[i];
    //}

    LOG_INFO("Calculating node metrics");
    
    int* ordering = (int *) read_ordering(ordering_path, max_node);
    NodeMetrics* metrics = ( NodeMetrics*) calloc(max_node + 1, sizeof(NodeMetrics));
    
    for (int node=1; node<=NUM_NODES; node++) {
        //int node = g->nodes[i];
        NodeMetrics* m = &metrics[node];
        
        m->out_degree = g->edges[node].count;
        for (int j = 0; j < g->edges[node].count; j++) {
            m->total_weight += g->edges[node].weights[j];
        }
        
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

double calculate_node_similarity(NodeMetrics m1, NodeMetrics m2) {
    double score = 5 * fabs(m1.in_degree - m2.in_degree) +
                   5 * fabs(m1.out_degree - m2.out_degree) +
                   fabs(m1.avg_in_weight - m2.avg_in_weight) +
                   fabs(m1.avg_out_weight - m2.avg_out_weight);
    
    double ordering_sim = fabs(m1.ordering_rank - m2.ordering_rank);
    score = 0.7 * score + 0.3 * ordering_sim;
    
    return -score;
}

int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int score = 0;
    
    for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
        //int src_m = gm->i;
        for (int j = 0; j < gm->edges[src_m].count; j++) {
            int dst_m = gm->edges[src_m].to_nodes[j];
            int weight_m = gm->edges[src_m].weights[j];
            int src_f = mapping[src_m];
            int dst_f = mapping[dst_m];
            score += MIN(weight_m, gf->adj_matrix[src_f][dst_f]);
        }
    }
    
    return score;
}

void validate_mapping_changes(int* old_mapping, int* new_mapping, int max_node,
                            int node_m1, int node_m2) {
    for (int i = 1; i <= max_node; i++) {
        if (i != node_m1 && i != node_m2) {
            if (old_mapping[i] != new_mapping[i]) {
                LOG_ERROR("Unexpected mapping change for node %d: %d -> %d",
                         i, old_mapping[i], new_mapping[i]);
            }
        }
    }
}

int calculate_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2) {
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int delta = 0;
    
    // Handle outgoing edges from node_m1
    for (int i = 0; i < gm->edges[node_m1].count; i++) {
        int dst_m = gm->edges[node_m1].to_nodes[i];
        if (dst_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m1].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int dst_m = gm->edges[node_m2].to_nodes[i];
        if (dst_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m2].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m2
    for (int i = 0; i < gm->reverse_edges[node_m2].count; i++) {
        int src_m = gm->reverse_edges[node_m2].to_nodes[i];
        if (src_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m2].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle direct edges between the swapped nodes
    // From m1 to m2
    int m1_to_m2 = gm->adj_matrix[node_m1][node_m2];
    if (m1_to_m2 > 0) {
        int old_weight = MIN(m1_to_m2, gf->adj_matrix[node_f1][node_f2]);
        int new_weight = MIN(m1_to_m2, gf->adj_matrix[node_f2][node_f1]);
        delta += new_weight - old_weight;
    }
    
    // From m2 to m1
    int m2_to_m1 = gm->adj_matrix[node_m2][node_m1];
    if (m2_to_m1 > 0) {
        int old_weight = MIN(m2_to_m1, gf->adj_matrix[node_f2][node_f1]);
        int new_weight = MIN(m2_to_m1, gf->adj_matrix[node_f1][node_f2]);
        delta += new_weight - old_weight;
    }
    
    return delta;
}

void write_mapping(const char* filename, int* mapping, int max_node) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        exit(1);
    }
    
    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 1; i <= max_node; i++) {
        if (mapping[i] != 0) {
            fprintf(file, "m%d,f%d\n", i, mapping[i]);
        }
    }
    
    fclose(file);
}

// Function to load benchmark mapping from CSV
int* load_benchmark_mapping(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    int* mapping = (int *) calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // Try to get expected score from filename
    int expected_score = 0;
    const char* underscore = strrchr(filename, '_');
    if (underscore) {
        expected_score = atoi(underscore + 1);
        LOG_INFO("Expected score from filename: %d", expected_score);
    }
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    LOG_INFO("Loading benchmark mapping from %s", filename);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int male_id, female_id;
        
        // Try direct integer format first
        if (sscanf(line, "%d,%d", &male_id, &female_id) == 2) {
            mapping[male_id] = female_id;
            count++;
        } else {
            // Try format with prefixes
            char male_str[20], female_str[20];
            if (sscanf(line, "%[^,],%s", male_str, female_str) == 2) {
                male_id = atoi(male_str + (male_str[0] == 'm' ? 1 : 0));
                female_id = atoi(female_str + (female_str[0] == 'f' ? 1 : 0));
                mapping[male_id] = female_id;
                count++;
            }
        }
    }
    
    //LOG_INFO("Loaded %s mappings", format_number(count));
    
    fclose(file);
    return mapping;
}

// Save intermediate mapping with verification
void save_intermediate_mapping(const char* filename, int* mapping, int max_node, 
                             Graph* gm, Graph* gf, int current_score) {
    write_mapping(filename, mapping, max_node);
    
    // Verify written mapping
    int* verification = load_benchmark_mapping(filename, max_node);
    if (verification) {
        int verify_score = calculate_alignment_score(gm, gf, verification);
        if (verify_score != current_score) {
            LOG_ERROR("Score mismatch - internal: %d, written: %d", 
                     current_score, verify_score);
        }
        free(verification);
    }
}

void random_swap_k_vertices(int* mapping, int n, int k,int seed) {
    // Create array for tracking selected vertices
    int* selected = (int*)malloc(2*k * sizeof(int));
    int* used = (int*)calloc(n, sizeof(int));  // Track used positions

    if (k*2>n) {
            LOG_ERROR("K is too big : %d",k); 
    } 
    // Randomly select k vertices
    int count = 0;
    srand(time(NULL)+seed);
    while (count < k*2) {
        int idx = rand() % n;
        if (!used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            count++;
        }
    }

    // Store original values
    int* original_values = (int*)malloc(2*k * sizeof(int));
    for (int i = 0; i <2* k; i++) {
        original_values[i] = mapping[selected[i]];

    }

    for (int i = 0; i < k; i++) {
        mapping[selected[i]] = original_values[i+k];
        mapping[selected[i+k]] = original_values[i];
    }

    // Clean up
    free(selected);
    free(used);
    free(original_values);
}




// Add this function
void verify_matrix_copy(Graph* g, int max_node) {
    short* h_adj_matrix = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));

    // Count non-zero entries in original
    int orig_nonzero = 0;
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            if (g->adj_matrix[i][j] > 0) {
                orig_nonzero++;
                printf("Found edge %d->%d weight=%d\n",
                       i, j, g->adj_matrix[i][j]);
            }
            h_adj_matrix[i * (max_node + 1) + j] = g->adj_matrix[i][j];
        }
    }

    // Verify copied matrix
    int copy_nonzero = 0;
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            if (h_adj_matrix[i * (max_node + 1) + j] > 0) {
                copy_nonzero++;
            }
        }
    }

    printf("Original matrix had %d edges, copied matrix has %d edges\n",
           orig_nonzero, copy_nonzero);

    free(h_adj_matrix);
}







// Add to optimize_mapping function
void apply_permutation(int* mapping, int node1, int node2, int node3, int perm_idx) {
    int f1 = mapping[node1];
    int f2 = mapping[node2];
    int f3 = mapping[node3];
    
    // Permutation mappings based on perm_idx
    switch(perm_idx) {
        case 1: // f1,f3,f2
            mapping[node2] = f3;
            mapping[node3] = f2;
            break;
        case 2: // f2,f1,f3
            mapping[node1] = f2;
            mapping[node2] = f1;
            break;
        case 3: // f2,f3,f1
            mapping[node1] = f2;
            mapping[node2] = f3;
            mapping[node3] = f1;
            break;
        case 4: // f3,f1,f2
            mapping[node1] = f3;
            mapping[node2] = f1;
            mapping[node3] = f2;
            break;
        case 5: // f3,f2,f1
            mapping[node1] = f3;
            mapping[node2] = f2;
            mapping[node3] = f1;
            break;
    }
}





// Modified optimize_mapping function
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping, const char* out_path) {




    // Check if CUDA is already initialized
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA not initialized: %s", cudaGetErrorString(err));
        return NULL;
    }
    printf("Using CUDA device %d\n", device);
    // Print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);






    int rank=0;
    int size=4;
    
    int max_node = NUM_NODES;
    
    // Allocate and initialize mappings
    int* current_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    int* best_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    int* old_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    memcpy(current_mapping, initial_mapping, (max_node + 1) * sizeof(int));
    memcpy(best_mapping, current_mapping, (max_node + 1) * sizeof(int));
    
    // Prepare GPU data structures
    short *d_adj_matrix_m, *d_adj_matrix_f;
    int *d_mapping, *d_deltas, *d_node_pairs;
    



    // Allocate GPU memory
    cudaMalloc(&d_adj_matrix_m, (max_node + 1) * (max_node + 1) * sizeof(short));
    cudaMalloc(&d_adj_matrix_f, (max_node + 1) * (max_node + 1) * sizeof(short));
    cudaMalloc(&d_mapping, (max_node + 1) * sizeof(int));
    
    // Convert adjacency matrices to linear arrays and copy to GPU
    short* h_adj_matrix_m = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));
    short* h_adj_matrix_f = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));
    
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            h_adj_matrix_m[i * (max_node + 1) + j] = gm->adj_matrix[i][j];
            h_adj_matrix_f[i * (max_node + 1) + j] = gf->adj_matrix[i][j];
        }
    }
    




    cudaMemcpy(d_adj_matrix_m, h_adj_matrix_m, (max_node + 1) * (max_node + 1) * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_matrix_f, h_adj_matrix_f, (max_node + 1) * (max_node + 1) * sizeof(short), cudaMemcpyHostToDevice);
    


    // In the main optimization loop:
    long int total_triplets = ((long int) max_node * (max_node - 1) * (max_node - 2)) / 6;
   
    printf("total triplets =%ld\n",total_triplets);
    //int batch_size = min(BATCH_SIZE, total_triplets);
    int batch_size = BATCH_SIZE;
    int* d_triplet_deltas;
    int* d_node_triplets;

    cudaMalloc(&d_triplet_deltas, batch_size * sizeof(int));
    cudaMalloc(&d_node_triplets, batch_size * 4 * sizeof(int));

    int* h_triplet_deltas = (int*)malloc(batch_size * sizeof(int));
    int* h_node_triplets = (int*)malloc(batch_size * 4 * sizeof(int));



    // Initialize other variables
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    int best_score = current_score;
    int improvements = 0;
    time_t start_time = time(NULL);
    time_t last_sync_time = start_time;
    int last = 0, pass = 0;
    int iter=0;
    
    // Apply initial random perturbation
    //random_swap_k_vertices(current_mapping, max_node, 10 + rank, rank);
    //current_score = calculate_alignment_score(gm, gf, current_mapping);
    

    LOG_INFO("before loop, current score=%d, best score =%d", current_score, best_score);
    // Main optimization loop
    while (true) {
        // Synchronization block
        time_t current_time = time(NULL);
        

        if (difftime(current_time, last_sync_time) >= SYNC_INTERVAL) {
              LOG_INFO("Process %d, enter synchronization the best score is %d", rank,best_score);
              int* benchmark_mapping = load_benchmark_mapping("../data/best.csv", max_node);
              if (!benchmark_mapping) {
                  LOG_ERROR("Failed to load benchmark mapping");
              }

              int benchmark_score = calculate_alignment_score(gm, gf, benchmark_mapping);
              if (benchmark_score >best_score) {
                      best_score=benchmark_score;
                      memcpy(best_mapping, benchmark_mapping, sizeof(int) * (max_node + 1));
              }
              free(benchmark_mapping);


              last_sync_time = current_time;
              LOG_INFO("Process %d, after synchronization and best score is %d", rank,best_score);

        }



        // Copy current mapping to GPU
        cudaMemcpy(d_mapping, current_mapping, (max_node + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        bool found_improvement = false;
        long int batch_start = 0;
    
        while (batch_start < total_triplets) {
	    long int restsize=total_triplets - batch_start;
            int current_batch_size = batch_size;
	    if (restsize <batch_size ) {
		    current_batch_size =(int) restsize;
	    }
        
            // Launch kernel for current batch
            int threadsPerBlock = 256;
            int numBlocks = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
        
            calculateThreeNodeSwapKernel<<<numBlocks, threadsPerBlock>>>(
                d_adj_matrix_m,
                d_adj_matrix_f,
                d_mapping,
                max_node,
                batch_start,
                current_batch_size,
                d_triplet_deltas,
                d_node_triplets
            );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA synchronize err: %s", cudaGetErrorString(err));
        return NULL;
    }


            // Copy results back
            err=cudaMemcpy(h_triplet_deltas, d_triplet_deltas, 
                   current_batch_size * sizeof(int), cudaMemcpyDeviceToHost);


    if (err != cudaSuccess) {
        LOG_ERROR("CUDA not initialized: %s", cudaGetErrorString(err));
        return NULL;
    }


            err=cudaMemcpy(h_node_triplets, d_node_triplets, 
                   current_batch_size * 4 * sizeof(int), cudaMemcpyDeviceToHost);
        
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA not initialized: %s", cudaGetErrorString(err));
        return NULL;
    }
            // Find best improvement in batch
            int best_delta = 0;
            int best_idx = -1;
        
            for (int i = 0; i < current_batch_size; i++) {
                if (h_triplet_deltas[i] > best_delta) {
                    best_delta = h_triplet_deltas[i];
                    best_idx = i;
                }
            }
        
            printf("max swap index =%d, max delta= %d from batch start %d\n",best_idx,best_delta,batch_start);
            // Apply best improvement if found
            if (best_delta > 0) {
                int node1 = h_node_triplets[best_idx * 4];
                int node2 = h_node_triplets[best_idx * 4 + 1];
                int node3 = h_node_triplets[best_idx * 4 + 2];
                int perm = h_node_triplets[best_idx * 4 + 3];
            
                apply_permutation(current_mapping, node1, node2, node3, perm);
                current_score += best_delta;
            
                if (current_score > best_score) {
                    memcpy(best_mapping, current_mapping, (max_node + 1) * sizeof(int));
                    best_score = current_score;
                    LOG_INFO("Process %d found new best score: %d", rank, best_score);
                    save_intermediate_mapping(out_path, best_mapping, max_node, gm, gf, best_score);
                }
            
                found_improvement = true;
		improvements++;
                break;  // Start over with new mapping
            }
        
            batch_start += (long int) current_batch_size + total_triplets/3;

        }
    
        // If no improvements found in any batch, do random perturbation
        if (!found_improvement) {
            memcpy(current_mapping, best_mapping, (max_node + 1) * sizeof(int));
            int num_pairs = 3 + rand() % 3;
            random_swap_k_vertices(current_mapping, max_node, num_pairs, rank);
            current_score = calculate_alignment_score(gm, gf, current_mapping);

            LOG_INFO("Process %d restarting optimization with perturbation pairs =%d, best score =%d", rank,num_pairs,best_score);
            LOG_INFO("after shuffle current_score=  %d", current_score);
	    improvements=0;
        }
        current_time = time(NULL);
	//if (iter++ % 2==0) {
	{   iter++;

            LOG_INFO("Average time for this loop is  %f for iteration %d", (current_time-start_time)/(iter*1.0),iter);
            LOG_INFO("Current score is %d  best score is %d of rank %d, improvement =%d", current_score, best_score, rank,improvements);
	}
    }
    
    // Cleanup
    cudaFree(d_adj_matrix_m);
    cudaFree(d_adj_matrix_f);
    cudaFree(d_mapping);
    cudaFree(d_triplet_deltas);
    cudaFree(d_node_triplets);
    


    free(h_triplet_deltas);
    free(h_node_triplets);
    free(h_adj_matrix_m);
    free(h_adj_matrix_f);
    free(current_mapping);
    free(old_mapping);
    
    return best_mapping;
}

// Function to get maximum node ID from graph
int get_max_node(Graph* g) {
    return NUM_NODES;
}

// Function to clean up graph memory
void free_graph(Graph* g) {
    if (g->edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
        }
        free(g->edges);
    }
    
    if (g->reverse_edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        free(g->reverse_edges);
    }
    if (g->adj_matrix){
        for (int i=0; i<=NUM_NODES; i++){
            free(g->adj_matrix[i]);
        }
        free(g->adj_matrix);
        g->adj_matrix=NULL;
    }
    //free(g->nodes);
    free(g);
}

Graph* load_graph_from_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    Graph* graph = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int total_lines = 0;
    
    // Count total lines for progress bar
    while (fgets(line, MAX_LINE_LENGTH, file)) total_lines++;
    rewind(file);
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    total_lines--; // Adjust for header
    
    LOG_INFO("Loading graph from %s (%s lines)", filename, format_number(total_lines));
    
    time_t start_time = time(NULL);
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(graph, from, to, weight);
            line_count++;
            if (line_count % 100000 == 0) {
                print_progress(line_count, total_lines, "Loading graph");
            }
        } else {
            LOG_ERROR("Malformed line in CSV: %s", line);
        }
    }
    
    time_t end_time = time(NULL);
    LOG_INFO("Graph loaded successfully:");
    LOG_INFO("  - Nodes: %s", format_number(NUM_NODES));
    LOG_INFO("  - Edges: %s", format_number(line_count));
    LOG_INFO("  - Time taken: %ld seconds", end_time - start_time);
    
    fclose(file);
    return graph;
}

// Main function
int main(int argc, char* argv[]) {



    // Initialize CUDA first
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed! Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print device info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);











    int rank=0;
    int size=4;

    if (argc < 5) {
        if (rank == 0) {
            LOG_ERROR("Usage: %s <male graph> <female graph> <in mapping> <out mapping>", argv[0]);
        }
        return 1;
    }
	

    time_t total_start = time(NULL);
    // Only rank 0 prints initial information
    if (rank == 0) {
        LOG_INFO("Graph Alignment Tool v1.0 (MPI + OpenMP)");
        LOG_INFO("Number of MPI processes: %d", size);
        LOG_INFO("Starting process with:");
        LOG_INFO("  - Male graph: %s", argv[1]);
        LOG_INFO("  - Female graph: %s", argv[2]);
        LOG_INFO("  - Output mapping: %s", argv[4]);
    }

    
    Graph* gm = load_graph_from_csv(argv[1]);
    if (!gm) {
        LOG_ERROR("Failed to load male graph");
        return 1;
    }
    
    Graph* gf = load_graph_from_csv(argv[2]);
    if (!gf) {
        LOG_ERROR("Failed to load female graph");
        free_graph(gm);
        return 1;
    }
    
    int max_node = MAX(get_max_node(gm), get_max_node(gf));
    
    int* benchmark = load_benchmark_mapping(argv[3], max_node);
    if (!benchmark) {
        LOG_ERROR("Failed to load benchmark mapping");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }
    
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    LOG_INFO("Initial alignment score: %s", format_number(initial_score));
    
    char outputfilename[100];

    sprintf(outputfilename, "ID%d-%s", rank,argv[4]);

    sprintf(outputfilename, "%.*s-rank%d.csv", (int)strlen(argv[4])-4, argv[4], rank);
    LOG_INFO("output file name is %s",outputfilename);

    int* optimized_mapping = optimize_mapping(gm, gf, benchmark, outputfilename);
    int optimized_score = calculate_alignment_score(gm, gf, optimized_mapping);
    
    time_t total_end = time(NULL);
    LOG_INFO("Process completed:");
    LOG_INFO("  - Initial score: %s", format_number(initial_score));
    LOG_INFO("  - Final score: %s", format_number(optimized_score));
    LOG_INFO("  - Improvement: %.2f%%",
            (double)(optimized_score - initial_score) / initial_score * 100.0);
    LOG_INFO("  - Total time: %.1f minutes", difftime(total_end, total_start) / 60);
    
    free_graph(gm);
    free_graph(gf);
    free(benchmark);
    free(optimized_mapping);
    
    return 0;
}
