#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INITIAL_NODE_CAPACITY 1024
#define HASH_TABLE_SIZE 16384  // Power of 2 for better hash distribution

// Hash table node structure
typedef struct NodeEntry {
    int node;
    struct NodeEntry *next;
} NodeEntry;

// Hash table for O(1) node lookups
typedef struct {
    NodeEntry **buckets;
    int size;
} HashTable;

// Create a new hash table
HashTable* create_hash_table(int size) {
    HashTable *table = malloc(sizeof(HashTable));
    table->size = size;
    table->buckets = calloc(size, sizeof(NodeEntry*));
    return table;
}

// Hash function
static inline int hash(int key, int size) {
    return abs(key) & (size - 1);  // Fast modulo for power of 2
}

// Check if node exists in hash table - O(1) average case
int node_exists(HashTable *table, int node) {
    int index = hash(node, table->size);
    NodeEntry *entry = table->buckets[index];
    
    while (entry != NULL) {
        if (entry->node == node) return 1;
        entry = entry->next;
    }
    return 0;
}

// Add node to hash table
void add_node(HashTable *table, int node) {
    int index = hash(node, table->size);
    
    // Create new entry
    NodeEntry *new_entry = malloc(sizeof(NodeEntry));
    new_entry->node = node;
    new_entry->next = table->buckets[index];
    table->buckets[index] = new_entry;
}

// Free hash table memory
void free_hash_table(HashTable *table) {
    for (int i = 0; i < table->size; i++) {
        NodeEntry *entry = table->buckets[i];
        while (entry != NULL) {
            NodeEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(table->buckets);
    free(table);
}

// Optimized function to load unique nodes
int *load_unique_nodes(const char *filename, int *num_nodes) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    // Initialize hash table for O(1) lookups
    HashTable *table = create_hash_table(HASH_TABLE_SIZE);
    
    // Initialize dynamic array for nodes
    int capacity = INITIAL_NODE_CAPACITY;
    int *nodes = malloc(capacity * sizeof(int));
    *num_nodes = 0;

    // Buffer for reading lines
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    // Skip header
    getline(&line, &len, file);

    // Process file using getline for arbitrary length lines
    while ((read = getline(&line, &len, file)) != -1) {
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) != 3) {
            continue;
        }

        // Process source node
        if (!node_exists(table, src)) {
            if (*num_nodes >= capacity) {
                capacity *= 2;
                int *temp = realloc(nodes, capacity * sizeof(int));
                if (!temp) {
                    free(nodes);
                    free_hash_table(table);
                    free(line);
                    fclose(file);
                    return NULL;
                }
                nodes = temp;
            }
            nodes[(*num_nodes)++] = src;
            add_node(table, src);
        }

        // Process target node
        if (!node_exists(table, tgt)) {
            if (*num_nodes >= capacity) {
                capacity *= 2;
                int *temp = realloc(nodes, capacity * sizeof(int));
                if (!temp) {
                    free(nodes);
                    free_hash_table(table);
                    free(line);
                    fclose(file);
                    return NULL;
                }
                nodes = temp;
            }
            nodes[(*num_nodes)++] = tgt;
            add_node(table, tgt);
        }
    }

    // Cleanup
    free(line);
    free_hash_table(table);
    fclose(file);

    // Shrink array to actual size
    if (*num_nodes > 0) {
        int *temp = realloc(nodes, *num_nodes * sizeof(int));
        return temp ? temp : nodes;
    }
    return nodes;
}

// Optimized function to write identity matching
void write_identity_matching(const char *filename, const int *nodes, int num_nodes) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    // Write header
    fputs("Node G1,Node G2\n", file);

    // Use buffer for better I/O performance
    char buffer[4096];
    size_t buffer_pos = 0;

    for (int i = 0; i < num_nodes; i++) {
        int written = snprintf(buffer + buffer_pos, sizeof(buffer) - buffer_pos, 
                             "%d,%d\n", nodes[i], nodes[i]);
        
        buffer_pos += written;

        // Flush buffer when it's nearly full
        if (sizeof(buffer) - buffer_pos < 32) {
            fwrite(buffer, 1, buffer_pos, file);
            buffer_pos = 0;
        }
    }

    // Write remaining buffer content
    if (buffer_pos > 0) {
        fwrite(buffer, 1, buffer_pos, file);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_graph.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_nodes;
    int *nodes = load_unique_nodes(argv[1], &num_nodes);
    if (!nodes) {
        fprintf(stderr, "Error loading unique nodes from %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    write_identity_matching(argv[2], nodes, num_nodes);
    free(nodes);
    
    printf("Identity matching created in %s\n", argv[2]);
    return EXIT_SUCCESS;
}
