#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapack.h>

#define INITIAL_NODE_CAPACITY 100
#define MAX_LINE_LENGTH 256

typedef struct {
    int *node_ids;        // Array to store original node IDs
    int *id_to_index;     // Mapping from node ID to matrix index
    int max_node_id;      // Maximum node ID encountered
    int num_nodes;        // Number of nodes in the graph
    double **adj_matrix;  // Adjacency matrix
} Graph;

// Function declarations
Graph *initialize_graph(int num_nodes, int max_node_id);
void free_graph(Graph *graph);
Graph *load_graph(const char *filename, int *num_nodes);
int compute_eigen_decomposition(Graph *graph, double *eigenvalues, double *eigenvectors);
void match_vertices(Graph *g1, Graph *g2, int *matching);
void write_matching(const char *filename, int *matching, Graph *g1, Graph *g2);

Graph *initialize_graph(int num_nodes, int max_node_id) {
    printf("Initializing graph with %d nodes\n", num_nodes);
    Graph *graph = malloc(sizeof(Graph));
    if (!graph) {
        fprintf(stderr, "Memory allocation failed for Graph structure\n");
        return NULL;
    }

    graph->num_nodes = num_nodes;
    graph->max_node_id = max_node_id;
    
    // Allocate node_ids array
    graph->node_ids = malloc(num_nodes * sizeof(int));
    if (!graph->node_ids) {
        fprintf(stderr, "Memory allocation failed for node_ids array\n");
        free(graph);
        return NULL;
    }

    // Allocate id_to_index mapping array
    graph->id_to_index = malloc((max_node_id + 1) * sizeof(int));
    if (!graph->id_to_index) {
        fprintf(stderr, "Memory allocation failed for id_to_index array\n");
        free(graph->node_ids);
        free(graph);
        return NULL;
    }
    
    // Initialize id_to_index with -1 to indicate unused indices
    for (int i = 0; i <= max_node_id; i++) {
        graph->id_to_index[i] = -1;
    }

    // Allocate adjacency matrix
    graph->adj_matrix = malloc(num_nodes * sizeof(double *));
    if (!graph->adj_matrix) {
        fprintf(stderr, "Memory allocation failed for adjacency matrix\n");
        free(graph->id_to_index);
        free(graph->node_ids);
        free(graph);
        return NULL;
    }

    for (int i = 0; i < num_nodes; i++) {
        graph->adj_matrix[i] = calloc(num_nodes, sizeof(double));
        if (!graph->adj_matrix[i]) {
            fprintf(stderr, "Memory allocation failed for adjacency matrix row %d\n", i);
            for (int j = 0; j < i; j++) {
                free(graph->adj_matrix[j]);
            }
            free(graph->adj_matrix);
            free(graph->id_to_index);
            free(graph->node_ids);
            free(graph);
            return NULL;
        }
    }

    return graph;
}

void free_graph(Graph *graph) {
    if (!graph) return;
    
    if (graph->adj_matrix) {
        for (int i = 0; i < graph->num_nodes; i++) {
            free(graph->adj_matrix[i]);
        }
        free(graph->adj_matrix);
    }
    
    free(graph->node_ids);
    free(graph->id_to_index);
    free(graph);
}

Graph *load_graph(const char *filename, int *num_nodes) {
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

    int node_count = 0;
    int max_node_id = -1;
    char line[MAX_LINE_LENGTH];
    
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Error reading header from file\n");
        free(temp_nodes);
        fclose(file);
        return NULL;
    }

    // First pass: collect unique nodes and find max node ID
    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) {
            fprintf(stderr, "Error parsing line: %s", line);
            continue;
        }

        max_node_id = src > max_node_id ? src : max_node_id;
        max_node_id = tgt > max_node_id ? tgt : max_node_id;

        // Add source node if not already present
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

        // Add target node if not already present
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

    *num_nodes = node_count;
    Graph *graph = initialize_graph(node_count, max_node_id);
    if (!graph) {
        free(temp_nodes);
        fclose(file);
        return NULL;
    }

    // Copy node IDs and create mapping
    memcpy(graph->node_ids, temp_nodes, node_count * sizeof(int));
    for (int i = 0; i < node_count; i++) {
        graph->id_to_index[temp_nodes[i]] = i;
    }

    // Second pass: fill adjacency matrix
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header again
    
    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) {
            continue;
        }

        int src_idx = graph->id_to_index[src];
        int tgt_idx = graph->id_to_index[tgt];

        if (src_idx != -1 && tgt_idx != -1) {
            graph->adj_matrix[src_idx][tgt_idx] = weight;
        }
    }

    free(temp_nodes);
    fclose(file);
    return graph;
}

// Forward declaration with const correctness matching LAPACK's expectations
extern void dgeev_(const char* jobvl, const char* jobvr, const int32_t* n,
                  double* a, const int32_t* lda, double* wr, double* wi,
                  double* vl, const int32_t* ldvl, double* vr,
                  const int32_t* ldvr, double* work, const int32_t* lwork,
                  int32_t* info, size_t jobvl_len, size_t jobvr_len);

int compute_eigen_decomposition(Graph *graph, double *eigenvalues, double *eigenvectors) {
    int32_t n = graph->num_nodes;
    const char jobvl = 'N';    // Don't compute left eigenvectors
    const char jobvr = 'V';    // Compute right eigenvectors
    int32_t lda = n;
    int32_t ldvl = 1;         // Leading dimension of vl (not used)
    int32_t ldvr = n;         // Leading dimension of vr
    int32_t info;
    int32_t lwork;
    double *wi = NULL;        // Imaginary parts of eigenvalues
    double *vl = NULL;        // Left eigenvectors (not computed)
    double *work = NULL;
    double *matrix = NULL;
    
    // Allocate memory for imaginary parts
    wi = malloc(n * sizeof(double));
    if (!wi) {
        fprintf(stderr, "Memory allocation failed for wi array\n");
        return -1;
    }

    // Allocate matrix in column-major order
    matrix = malloc(n * n * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed for matrix\n");
        free(wi);
        return -1;
    }

    // Copy adjacency matrix to column-major order
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[j * n + i] = graph->adj_matrix[i][j];
            if (!isfinite(matrix[j * n + i])) {
                fprintf(stderr, "Error: matrix[%d][%d] contains NaN or Infinity\n", i, j);
                free(matrix);
                free(wi);
                return -1;
            }
        }
    }

    // Query optimal work array size
    double worksize;
    lwork = -1;
    dgeev_(&jobvl, &jobvr, &n, matrix, &lda, eigenvalues, wi,
           vl, &ldvl, eigenvectors, &ldvr, &worksize, &lwork, &info, 1, 1);

    if (info != 0) {
        fprintf(stderr, "Work size query failed with error %d\n", info);
        free(matrix);
        free(wi);
        return -1;
    }

    lwork = (int32_t)worksize;
    work = malloc(lwork * sizeof(double));
    if (!work) {
        fprintf(stderr, "Memory allocation failed for work array\n");
        free(matrix);
        free(wi);
        return -1;
    }

    // Compute eigenvalues and eigenvectors
    dgeev_(&jobvl, &jobvr, &n, matrix, &lda, eigenvalues, wi,
           vl, &ldvl, eigenvectors, &ldvr, work, &lwork, &info, 1, 1);

    // Clean up
    free(work);
    free(matrix);
    free(wi);

    if (info > 0) {
        fprintf(stderr, "Error: dgeev failed to compute eigenvalues\n");
        return -1;
    } else if (info < 0) {
        fprintf(stderr, "Error: argument %d had an illegal value\n", -info);
        return -1;
    }

    return 0;
}


void match_vertices(Graph *g1, Graph *g2, int *matching) {
    int num_nodes = g1->num_nodes;
    
    // Allocate memory for eigenvalues and eigenvectors
    double *eigenvalues_g1 = aligned_alloc(64, num_nodes * sizeof(double));
    double *eigenvectors_g1 = aligned_alloc(64, num_nodes * num_nodes * sizeof(double));
    double *eigenvalues_g2 = aligned_alloc(64, num_nodes * sizeof(double));
    double *eigenvectors_g2 = aligned_alloc(64, num_nodes * num_nodes * sizeof(double));

    if (!eigenvalues_g1 || !eigenvectors_g1 || !eigenvalues_g2 || !eigenvectors_g2) {
        fprintf(stderr, "Memory allocation failed for eigen arrays\n");
        goto cleanup;
    }

    // Compute eigen decomposition for both graphs
    if (compute_eigen_decomposition(g1, eigenvalues_g1, eigenvectors_g1) < 0 ||
        compute_eigen_decomposition(g2, eigenvalues_g2, eigenvectors_g2) < 0) {
        goto cleanup;
    }

    // Match vertices based on eigenvector similarity
    for (int i = 0; i < num_nodes; i++) {
        int best_match = 0;
        double min_diff = INFINITY;
        
        for (int j = 0; j < num_nodes; j++) {
            double diff = 0.0;
            for (int k = 0; k < num_nodes; k++) {
                diff += fabs(eigenvectors_g1[i * num_nodes + k] - 
                           eigenvectors_g2[j * num_nodes + k]);
            }
            if (diff < min_diff) {
                min_diff = diff;
                best_match = j;
            }
        }
        matching[i] = g2->node_ids[best_match];
    }

cleanup:
    free(eigenvalues_g1);
    free(eigenvectors_g1);
    free(eigenvalues_g2);
    free(eigenvectors_g2);
}

void write_matching(const char *filename, int *matching, Graph *g1, Graph *g2) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < g1->num_nodes; i++) {
        fprintf(file, "%d,%d\n", g1->node_ids[i], matching[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_nodes1, num_nodes2;
    Graph *g1 = load_graph(argv[1], &num_nodes1);
    if (!g1) return EXIT_FAILURE;

    Graph *g2 = load_graph(argv[2], &num_nodes2);
    if (!g2) {
        free_graph(g1);
        return EXIT_FAILURE;
    }

    // Verify graphs have same number of nodes
    if (num_nodes1 != num_nodes2) {
        fprintf(stderr, "Error: Graphs have different numbers of nodes (%d vs %d)\n",
                num_nodes1, num_nodes2);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    int *matching = malloc(num_nodes1 * sizeof(int));
    if (!matching) {
        fprintf(stderr, "Memory allocation failed for matching array\n");
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    match_vertices(g1, g2, matching);
    write_matching(argv[3], matching, g1, g2);

    free(matching);
    free_graph(g1);
    free_graph(g2);

    return EXIT_SUCCESS;
}
