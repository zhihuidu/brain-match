#include <stdio.h>
#include <stdlib.h>

#define INITIAL_NODE_CAPACITY 1024

// Function to check if a node is already in the array of unique nodes
int node_exists(int *nodes, int num_nodes, int node) {
    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i] == node) {
            return 1;
        }
    }
    return 0;
}

// Function to load unique nodes from a graph CSV file
int *load_unique_nodes(const char *filename, int *num_nodes) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    int *nodes = malloc(INITIAL_NODE_CAPACITY * sizeof(int));
    int capacity = INITIAL_NODE_CAPACITY;
    *num_nodes = 0;

    char line[256];
    fgets(line, sizeof(line), file); // Skip header

    // Read each edge and add unique nodes
    while (fgets(line, sizeof(line), file)) {
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) != 3) {
            fprintf(stderr, "Error reading line: %s\n", line);
            continue;
        }

        // Add source node if it's unique
        if (!node_exists(nodes, *num_nodes, src)) {
            if (*num_nodes >= capacity) {
                capacity *= 2;
                nodes = realloc(nodes, capacity * sizeof(int));
                if (!nodes) {
                    fprintf(stderr, "Memory allocation error\n");
                    fclose(file);
                    return NULL;
                }
            }
            nodes[(*num_nodes)++] = src;
        }

        // Add target node if it's unique
        if (!node_exists(nodes, *num_nodes, tgt)) {
            if (*num_nodes >= capacity) {
                capacity *= 2;
                nodes = realloc(nodes, capacity * sizeof(int));
                if (!nodes) {
                    fprintf(stderr, "Memory allocation error\n");
                    fclose(file);
                    return NULL;
                }
            }
            nodes[(*num_nodes)++] = tgt;
        }
    }

    fclose(file);
    return nodes;
}

// Function to write identity matching to CSV
void write_identity_matching(const char *filename, int *nodes, int num_nodes) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(file, "%d,%d\n", nodes[i], nodes[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_graph.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_filename = argv[1];
    const char *matching_filename = argv[2];

    // Load unique nodes from the graph file
    int num_nodes;
    int *nodes = load_unique_nodes(graph_filename, &num_nodes);
    if (!nodes) {
        fprintf(stderr, "Error loading unique nodes from %s\n", graph_filename);
        return EXIT_FAILURE;
    }

    // Write identity matching to the output file
    write_identity_matching(matching_filename, nodes, num_nodes);

    // Free allocated memory
    free(nodes);

    printf("Identity matching created in %s\n", matching_filename);
    return EXIT_SUCCESS;
}
