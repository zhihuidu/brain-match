#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define INITIAL_NODE_CAPACITY 100

typedef struct {
    int *nodes;
    int num_nodes;
    double **adj_matrix; // Updated to double for eigen computation
} Graph;

// Function declarations (forward declarations)
Graph *initialize_graph(int num_nodes);
void free_graph(Graph *graph);
Graph *load_graph(const char *filename, int *num_nodes);
void compute_eigen_decomposition(Graph *graph, double *eigenvalues, double *eigenvectors);
void match_vertices(Graph *g1, Graph *g2, int *matching, int num_nodes);
void write_matching(const char *filename, int *matching, int num_nodes);

// Function to initialize the adjacency matrix
Graph *initialize_graph(int num_nodes) {
    printf("Initializing graph with %d nodes\n", num_nodes);
    Graph *graph = malloc(sizeof(Graph));
    if (!graph) {
        fprintf(stderr, "Memory allocation failed for Graph structure\n");
        exit(EXIT_FAILURE);
    }
    graph->num_nodes = num_nodes;
    graph->nodes = malloc(num_nodes * sizeof(int));
    if (!graph->nodes) {
        fprintf(stderr, "Memory allocation failed for nodes array\n");
        free(graph);
        exit(EXIT_FAILURE);
    }
    graph->adj_matrix = malloc(num_nodes * sizeof(double *));
    for (int i = 0; i < num_nodes; i++) {
        graph->adj_matrix[i] = calloc(num_nodes, sizeof(double));
        if (!graph->adj_matrix[i]) {
            fprintf(stderr, "Memory allocation failed for adjacency matrix\n");
            exit(EXIT_FAILURE);
        }
    }
    return graph;
}

// Function to free the graph's memory
void free_graph(Graph *graph) {
    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->adj_matrix[i]);
    }
    free(graph->adj_matrix);
    free(graph->nodes);
    free(graph);
}

// Function to load graph from CSV and fill adjacency matrix
Graph *load_graph(const char *filename, int *num_nodes) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    int capacity = INITIAL_NODE_CAPACITY;
    int *nodes = malloc(capacity * sizeof(int));
    int node_count = 0;
    char line[256];
    fgets(line, sizeof(line), file); // Skip header

    printf("Reading edges from %s\n", filename);

    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) {
            fprintf(stderr, "Error parsing line: %s\n", line);
            continue;
        }

        // Ensure node array has enough capacity
        if (node_count >= capacity) {
            capacity *= 2;
            nodes = realloc(nodes, capacity * sizeof(int));
            if (!nodes) {
                fprintf(stderr, "Memory allocation failed for nodes array\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }

        // Add source and target nodes to the nodes array if not already present
        int src_found = 0, tgt_found = 0;
        for (int i = 0; i < node_count; i++) {
            if (nodes[i] == src) src_found = 1;
            if (nodes[i] == tgt) tgt_found = 1;
        }
        if (!src_found) nodes[node_count++] = src;
        if (!tgt_found) nodes[node_count++] = tgt;

        printf("Parsed edge: src=%d, tgt=%d, weight=%f\n", src, tgt, weight);
    }

    *num_nodes = node_count;
    Graph *graph = initialize_graph(node_count);

    // Fill adjacency matrix
    fseek(file, 0, SEEK_SET); // Reset file pointer
    fgets(line, sizeof(line), file); // Skip header again
    while (fgets(line, sizeof(line), file)) {
        int src, tgt;
        double weight;
        if (sscanf(line, "%d,%d,%lf", &src, &tgt, &weight) != 3) {
            fprintf(stderr, "Error parsing line: %s\n", line);
            continue;
        }

        int src_idx = src % node_count;
        int tgt_idx = tgt % node_count;

        if (src_idx < 0 || src_idx >= node_count || tgt_idx < 0 || tgt_idx >= node_count) {
            fprintf(stderr, "Error: Index out of bounds in adjacency matrix for src=%d, tgt=%d\n", src, tgt);
            fclose(file);
            free(nodes);
            free_graph(graph);
            exit(EXIT_FAILURE);
        }

        graph->adj_matrix[src_idx][tgt_idx] = weight;
        printf("Setting matrix[%d][%d] = %f\n", src_idx, tgt_idx, weight);
    }

    fclose(file);
    free(nodes);
    return graph;
}


// Function to compute eigenvalues and eigenvectors using LAPACK in column-major order
void compute_eigen_decomposition(Graph *graph, double *eigenvalues, double *eigenvectors) {
    int n = graph->num_nodes;

    // Allocate aligned memory for the matrix with 64-byte alignment
    double *matrix = aligned_alloc(64, n * n * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed for matrix\n");
        exit(EXIT_FAILURE);
    }

    printf("Copying adjacency matrix to LAPACK-compatible array in column-major order for %d nodes\n", n);

    // Copy adjacency matrix to matrix in column-major order and check values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[j * n + i] = graph->adj_matrix[i][j];

            // Validate that each matrix element is finite
            if (!isfinite(matrix[j * n + i])) {
                fprintf(stderr, "Error: matrix[%d][%d] contains NaN or Infinity: %f\n", i, j, matrix[j * n + i]);
                free(matrix);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Allocate aligned memory for eigenvalues and eigenvectors if they haven’t been allocated
    eigenvalues = aligned_alloc(64, n * sizeof(double));
    eigenvectors = aligned_alloc(64, n * n * sizeof(double));
    if (!eigenvalues || !eigenvectors) {
        fprintf(stderr, "Memory allocation failed for eigenvalues or eigenvectors\n");
        free(matrix);
        free(eigenvalues);
        free(eigenvectors);
        exit(EXIT_FAILURE);
    }

    // Initialize eigenvalues and eigenvectors
    for (int i = 0; i < n; i++) eigenvalues[i] = 0.0;
    for (int i = 0; i < n * n; i++) eigenvectors[i] = 0.0;

    int lda = n;
    int info;

    printf("Calling LAPACKE_dgeev for general eigen decomposition\n");

    // Perform eigen decomposition
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n, matrix, lda, eigenvalues, NULL, NULL, lda, eigenvectors, lda);
    if (info > 0) {
        fprintf(stderr, "Error: LAPACKE_dgeev failed to compute eigenvalues, info=%d\n", info);
        free(matrix);
        free(eigenvalues);
        free(eigenvectors);
        exit(EXIT_FAILURE);
    }

    printf("Eigen decomposition completed successfully\n");

    free(matrix);
}


void match_vertices(Graph *g1, Graph *g2, int *matching, int num_nodes) {
    printf("Allocating memory for eigenvalues and eigenvectors\n");

    double *eigenvalues_g1 = aligned_alloc(64, num_nodes * sizeof(double));
    double *eigenvectors_g1 = aligned_alloc(64, num_nodes * num_nodes * sizeof(double));

    double *eigenvalues_g2 = aligned_alloc(64, num_nodes * sizeof(double));
    double *eigenvectors_g2 = aligned_alloc(64, num_nodes * num_nodes * sizeof(double));

    if (!eigenvalues_g1 || !eigenvectors_g1 || !eigenvalues_g2 || !eigenvectors_g2) {
        fprintf(stderr, "Memory allocation failed for eigenvalue/eigenvector arrays\n");
        free(eigenvalues_g1);
        free(eigenvectors_g1);
        free(eigenvalues_g2);
        free(eigenvectors_g2);
        exit(EXIT_FAILURE);
    }

    // Initialize eigenvector arrays to prevent any uninitialized memory usage
    for (int i = 0; i < num_nodes * num_nodes; i++) {
        eigenvectors_g1[i] = 0.0;
        eigenvectors_g2[i] = 0.0;
    }

    printf("Computing eigen decomposition for Graph 1\n");
    compute_eigen_decomposition(g1, eigenvalues_g1, eigenvectors_g1);

    printf("Computing eigen decomposition for Graph 2\n");
    compute_eigen_decomposition(g2, eigenvalues_g2, eigenvectors_g2);

    printf("Matching vertices based on eigenvector similarity\n");

    for (int i = 0; i < num_nodes; i++) {
        int best_match = 0;
        double min_diff = INFINITY;
        for (int j = 0; j < num_nodes; j++) {
            double diff = 0.0;
            for (int k = 0; k < num_nodes; k++) {
                diff += fabs(eigenvectors_g1[i * num_nodes + k] - eigenvectors_g2[j * num_nodes + k]);
            }
            if (diff < min_diff) {
                min_diff = diff;
                best_match = j;
            }
        }
        matching[i] = best_match;
    }

    printf("Vertex matching completed\n");

    free(eigenvalues_g1);
    free(eigenvectors_g1);
    free(eigenvalues_g2);
    free(eigenvectors_g2);
}

// Function to write matching to CSV
void write_matching(const char *filename, int *matching, int num_nodes) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(file, "%d,%d\n", i, matching[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *g1_filename = argv[1];
    const char *g2_filename = argv[2];
    const char *output_filename = argv[3];

    int num_nodes;
    printf("Loading graph 1 from %s\n", g1_filename);
    Graph *g1 = load_graph(g1_filename, &num_nodes);
    if (!g1) {
        return EXIT_FAILURE;
    }
    printf("Graph 1 loaded with %d nodes\n", num_nodes);

    printf("Loading graph 2 from %s\n", g2_filename);
    Graph *g2 = load_graph(g2_filename, &num_nodes);
    if (!g2) {
        free_graph(g1);
        return EXIT_FAILURE;
    }
    printf("Graph 2 loaded with %d nodes\n", num_nodes);

    int *matching = malloc(num_nodes * sizeof(int));
    if (!matching) {
        fprintf(stderr, "Memory allocation error for matching array.\n");
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }

    printf("Matching vertices based on spectral properties\n");
    match_vertices(g1, g2, matching, num_nodes);
    write_matching(output_filename, matching, num_nodes);

    printf("Refined matching saved to %s\n", output_filename);

    free(matching);
    free_graph(g1);
    free_graph(g2);

    return EXIT_SUCCESS;
}
