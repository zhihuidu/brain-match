#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

// Error codes
#define SUCCESS 0
#define ERROR_FILE_OPEN -1
#define ERROR_MEMORY_ALLOC -2
#define ERROR_INVALID_INPUT -3

// Initial sizes and thresholds
#define INITIAL_HASH_SIZE 1024
#define INITIAL_GRAPH_SIZE 1024
#define LOAD_FACTOR_THRESHOLD 0.7
#define PROGRESS_INTERVAL 10000

// Forward declarations of all structures
typedef struct EdgeNode EdgeNode;
typedef struct Graph Graph;
typedef struct HashNode HashNode;
typedef struct HashMap HashMap;
typedef struct VertexMap VertexMap;

// Structure definitions
struct EdgeNode {
    int target;
    int weight;
    struct EdgeNode* next;
};

struct Graph {
    EdgeNode** adjacency_lists;
    size_t num_vertices;
    size_t num_edges;
};

struct HashNode {
    int key;
    int value;
    struct HashNode* next;
};

struct HashMap {
    HashNode** buckets;
    size_t size;
    size_t count;
};

struct VertexMap {
    HashMap* id_to_index;
    int* index_to_id;
    size_t count;
    size_t capacity;
};

// Function prototypes
void print_timestamp(const char* message);
void print_progress(size_t current, size_t total, const char* task);
void print_memory_usage(const Graph* g1, const Graph* g2, const VertexMap* vmap, const char* stage);

unsigned int hash_function(int key);
HashMap* create_hashmap(size_t initial_size);
void free_hashmap(HashMap* map);
int resize_hashmap(HashMap* map);
int hashmap_put(HashMap* map, int key, int value);
int hashmap_get(const HashMap* map, int key, int* value);

VertexMap* create_vertex_map(size_t initial_capacity);
void free_vertex_map(VertexMap* vmap);
int get_or_create_vertex_index(VertexMap* vmap, int original_id);
int get_original_id(const VertexMap* vmap, int index);

Graph* create_graph(size_t initial_vertices);
void free_graph(Graph* graph);
int add_edge(Graph* graph, int source, int target, int weight);
EdgeNode* find_edge(const Graph* graph, int source, int target);
int load_graph_sparse(const char* filename, Graph* graph, VertexMap* vmap);
int load_matching(const char* filename, VertexMap* vmap, int* matching, size_t matching_size);
int calculate_score_sparse(const Graph* g1, const Graph* g2, const int* matching);

// Progress reporting functions
void print_timestamp(const char* message) {
    time_t now;
    time(&now);
    char* time_str = ctime(&now);
    time_str[strlen(time_str) - 1] = '\0';
    fprintf(stderr, "[%s] %s\n", time_str, message);
}

void print_progress(size_t current, size_t total, const char* task) {
    if (total == 0) return;
    float percentage = (float)current * 100 / total;
    fprintf(stderr, "\r%s: %.1f%% (%zu/%zu)", task, percentage, current, total);
    if (current == total) fprintf(stderr, "\n");
    fflush(stderr);
}

void print_memory_usage(const Graph* g1, const Graph* g2, const VertexMap* vmap, const char* stage) {
    fprintf(stderr, "\nMemory usage after %s:\n", stage);
    fprintf(stderr, "  Graph 1: %zu vertices, %zu edges\n", 
            g1 ? g1->num_vertices : 0, g1 ? g1->num_edges : 0);
    fprintf(stderr, "  Graph 2: %zu vertices, %zu edges\n", 
            g2 ? g2->num_vertices : 0, g2 ? g2->num_edges : 0);
    fprintf(stderr, "  Vertex map: %zu vertices mapped\n\n", 
            vmap ? vmap->count : 0);
}

// Hash function implementation
unsigned int hash_function(int key) {
    unsigned int hash = 0;
    unsigned char* bytes = (unsigned char*)&key;
    
    for(size_t i = 0; i < sizeof(int); ++i) {
        hash += bytes[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    
    return hash;
}

// HashMap implementations
HashMap* create_hashmap(size_t initial_size) {
    HashMap* map = calloc(1, sizeof(HashMap));
    if (!map) return NULL;
    
    map->size = initial_size;
    map->buckets = calloc(initial_size, sizeof(HashNode*));
    if (!map->buckets) {
        free(map);
        return NULL;
    }
    
    return map;
}

void free_hashmap(HashMap* map) {
    if (!map) return;
    
    for (size_t i = 0; i < map->size; i++) {
        HashNode* current = map->buckets[i];
        while (current) {
            HashNode* next = current->next;
            free(current);
            current = next;
        }
    }
    
    free(map->buckets);
    free(map);
}

int resize_hashmap(HashMap* map) {
    size_t new_size = map->size * 2;
    HashNode** new_buckets = calloc(new_size, sizeof(HashNode*));
    if (!new_buckets) return ERROR_MEMORY_ALLOC;
    
    for (size_t i = 0; i < map->size; i++) {
        HashNode* current = map->buckets[i];
        while (current) {
            HashNode* next = current->next;
            unsigned int index = hash_function(current->key) & (new_size - 1);
            current->next = new_buckets[index];
            new_buckets[index] = current;
            current = next;
        }
    }
    
    free(map->buckets);
    map->buckets = new_buckets;
    map->size = new_size;
    
    return SUCCESS;
}

int hashmap_put(HashMap* map, int key, int value) {
    if (!map) return ERROR_INVALID_INPUT;
    
    if ((float)map->count / map->size >= LOAD_FACTOR_THRESHOLD) {
        if (resize_hashmap(map) != SUCCESS) return ERROR_MEMORY_ALLOC;
    }
    
    unsigned int index = hash_function(key) & (map->size - 1);
    
    HashNode* current = map->buckets[index];
    while (current) {
        if (current->key == key) {
            current->value = value;
            return SUCCESS;
        }
        current = current->next;
    }
    
    HashNode* node = malloc(sizeof(HashNode));
    if (!node) return ERROR_MEMORY_ALLOC;
    
    node->key = key;
    node->value = value;
    node->next = map->buckets[index];
    map->buckets[index] = node;
    map->count++;
    
    return SUCCESS;
}

int hashmap_get(const HashMap* map, int key, int* value) {
    if (!map || !value) return ERROR_INVALID_INPUT;
    
    unsigned int index = hash_function(key) & (map->size - 1);
    HashNode* current = map->buckets[index];
    
    while (current) {
        if (current->key == key) {
            *value = current->value;
            return SUCCESS;
        }
        current = current->next;
    }
    
    return ERROR_INVALID_INPUT;
}

// VertexMap implementations
VertexMap* create_vertex_map(size_t initial_capacity) {
    VertexMap* vmap = calloc(1, sizeof(VertexMap));
    if (!vmap) return NULL;
    
    vmap->id_to_index = create_hashmap(initial_capacity);
    if (!vmap->id_to_index) {
        free(vmap);
        return NULL;
    }
    
    vmap->index_to_id = malloc(initial_capacity * sizeof(int));
    if (!vmap->index_to_id) {
        free_hashmap(vmap->id_to_index);
        free(vmap);
        return NULL;
    }
    
    vmap->capacity = initial_capacity;
    vmap->count = 0;
    return vmap;
}

void free_vertex_map(VertexMap* vmap) {
    if (vmap) {
        free_hashmap(vmap->id_to_index);
        free(vmap->index_to_id);
        free(vmap);
    }
}

int get_or_create_vertex_index(VertexMap* vmap, int original_id) {
    if (!vmap) return -1;
    
    int index;
    if (hashmap_get(vmap->id_to_index, original_id, &index) == SUCCESS) {
        return index;
    }
    
    if (vmap->count >= vmap->capacity) {
        size_t new_capacity = vmap->capacity * 2;
        int* new_array = realloc(vmap->index_to_id, new_capacity * sizeof(int));
        if (!new_array) return -1;
        
        vmap->index_to_id = new_array;
        vmap->capacity = new_capacity;
    }
    
    index = vmap->count;
    if (hashmap_put(vmap->id_to_index, original_id, index) != SUCCESS) {
        return -1;
    }
    
    vmap->index_to_id[index] = original_id;
    vmap->count++;
    return index;
}

int get_original_id(const VertexMap* vmap, int index) {
    if (!vmap || index < 0 || index >= vmap->count) return -1;
    return vmap->index_to_id[index];
}

// Graph implementations
Graph* create_graph(size_t initial_vertices) {
    Graph* graph = calloc(1, sizeof(Graph));
    if (!graph) return NULL;
    
    graph->adjacency_lists = calloc(initial_vertices, sizeof(EdgeNode*));
    if (!graph->adjacency_lists) {
        free(graph);
        return NULL;
    }
    
    graph->num_vertices = initial_vertices;
    graph->num_edges = 0;
    return graph;
}

void free_graph(Graph* graph) {
    if (!graph) return;
    
    for (size_t i = 0; i < graph->num_vertices; i++) {
        EdgeNode* current = graph->adjacency_lists[i];
        while (current) {
            EdgeNode* next = current->next;
            free(current);
            current = next;
        }
    }
    
    free(graph->adjacency_lists);
    free(graph);
}

EdgeNode* find_edge(const Graph* graph, int source, int target) {
    if (!graph || source >= graph->num_vertices) return NULL;
    
    EdgeNode* current = graph->adjacency_lists[source];
    while (current) {
        if (current->target == target) return current;
        current = current->next;
    }
    
    return NULL;
}

int add_edge(Graph* graph, int source, int target, int weight) {
    if (!graph || source >= graph->num_vertices || target >= graph->num_vertices) 
        return ERROR_INVALID_INPUT;
    
    EdgeNode* existing = find_edge(graph, source, target);
    if (existing) {
        existing->weight = weight;
        return SUCCESS;
    }
    
    EdgeNode* edge = malloc(sizeof(EdgeNode));
    if (!edge) return ERROR_MEMORY_ALLOC;
    
    edge->target = target;
    edge->weight = weight;
    edge->next = graph->adjacency_lists[source];
    graph->adjacency_lists[source] = edge;
    graph->num_edges++;
    
    return SUCCESS;
}

int load_graph_sparse(const char* filename, Graph* graph, VertexMap* vmap) {
    print_timestamp("Loading graph");
    fprintf(stderr, "Reading from file: %s\n", filename);
    
    FILE* file = fopen(filename, "r");
    if (!file) return ERROR_FILE_OPEN;
    
    char line[1024];
    size_t line_count = 0;
    size_t edge_count = 0;
    
    while (fgets(line, sizeof(line), file)) line_count++;
    rewind(file);
    
    fprintf(stderr, "Found %zu lines in file\n", line_count);
    
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return ERROR_INVALID_INPUT;
    }
    line_count--;
    
    size_t processed = 0;
    while (fgets(line, sizeof(line), file)) {
        processed++;
        if (processed % PROGRESS_INTERVAL == 0) {
            print_progress(processed, line_count, "Loading graph edges");
        }
        
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) != 3) continue;
        
        int src_idx = get_or_create_vertex_index(vmap, src);
        int tgt_idx = get_or_create_vertex_index(vmap, tgt);
        
        if (src_idx < 0 || tgt_idx < 0) {
            fclose(file);
            return ERROR_MEMORY_ALLOC;
        }
        
        if (src_idx >= graph->num_vertices || tgt_idx >= graph->num_vertices) {
            size_t new_size = (src_idx > tgt_idx ? src_idx : tgt_idx) + 1;
            EdgeNode** new_lists = realloc(graph->adjacency_lists, 
                                         new_size * sizeof(EdgeNode*));
            if (!new_lists) {
                fclose(file);
                return ERROR_MEMORY_ALLOC;
            }
            
            for (size_t i = graph->num_vertices; i < new_size; i++) {
                new_lists[i] = NULL;
            }
            
            graph->adjacency_lists = new_lists;
            graph->num_vertices = new_size;
        }
        
        if (add_edge(graph, src_idx, tgt_idx, weight) == SUCCESS) {
            edge_count++;
        }
    }
    
    fclose(file);
    fprintf(stderr, "\nSuccessfully loaded %zu edges\n", edge_count);
    return SUCCESS;
}

int load_matching(const char* filename, VertexMap* vmap, int* matching, size_t matching_size) {
    print_timestamp("Loading matching");
    fprintf(stderr, "Reading matching from file: %s\n", filename);
    
    FILE* file = fopen(filename, "r");
    if (!file) return ERROR_FILE_OPEN;
    
    // Initialize matching array
    for (size_t i = 0; i < matching_size; i++) {
        matching[i] = -1;
    }
    
    char line[1024];
    size_t line_count = 0;
    size_t matches = 0;
    
    // Count lines for progress reporting
    while (fgets(line, sizeof(line), file)) line_count++;
    rewind(file);
    
    fprintf(stderr, "Found %zu lines in matching file\n", line_count);
    
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return ERROR_INVALID_INPUT;
    }
    line_count--;
    
    size_t processed = 0;
    while (fgets(line, sizeof(line), file)) {
        processed++;
        if (processed % PROGRESS_INTERVAL == 0) {
            print_progress(processed, line_count, "Loading matches");
        }
        
        int id1, id2;
        if (sscanf(line, "%d,%d", &id1, &id2) != 2) continue;
        
        int idx1 = get_or_create_vertex_index(vmap, id1);
        int idx2 = get_or_create_vertex_index(vmap, id2);
        
        if (idx1 >= 0 && idx2 >= 0 && idx1 < matching_size) {
            matching[idx1] = idx2;
            matches++;
        }
    }
    
    fclose(file);
    fprintf(stderr, "\nSuccessfully loaded %zu matches\n", matches);
    return SUCCESS;
}

int calculate_score_sparse(const Graph* g1, const Graph* g2, const int* matching) {
    print_timestamp("Calculating matching alignment score");
    
    if (!g1 || !g2 || !matching) return -1;
    
    int alignment_score = 0;
    size_t processed_edges = 0;
    size_t total_edges = g1->num_edges; // Only process g1 edges
    
    // Process edges from g1 only (equivalent to male_edges in Python)
    for (size_t i = 0; i < g1->num_vertices; i++) {
        EdgeNode* edge1 = g1->adjacency_lists[i];
        while (edge1) {
            processed_edges++;
            if (processed_edges % PROGRESS_INTERVAL == 0) {
                print_progress(processed_edges, total_edges, "Computing alignment");
            }
            
            int source_g2 = matching[i];
            int target_g2 = matching[edge1->target];
            
            if (source_g2 != -1 && target_g2 != -1) {
                // Find corresponding edge in g2 (female_edges in Python)
                EdgeNode* edge2 = find_edge(g2, source_g2, target_g2);
                
                if (edge2) {
                    // Take minimum of the two edge weights
                    alignment_score += (edge1->weight < edge2->weight) ? 
                                     edge1->weight : edge2->weight;
                }
                // If edge doesn't exist in g2, contribute 0 (implicit in Python's .get())
            }
            
            edge1 = edge1->next;
        }
    }
    
    print_timestamp("Alignment score calculation complete");
    return alignment_score;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    print_timestamp("Starting graph matching program");
    
    const char* g1_filename = argv[1];
    const char* g2_filename = argv[2];
    const char* matching_filename = argv[3];
    
    // Create vertex mapping
    VertexMap* vmap = create_vertex_map(INITIAL_HASH_SIZE);
    if (!vmap) {
        fprintf(stderr, "Failed to create vertex map\n");
        return EXIT_FAILURE;
    }
    
    // Create initial empty graphs
    Graph* g1 = create_graph(INITIAL_GRAPH_SIZE);
    Graph* g2 = create_graph(INITIAL_GRAPH_SIZE);
    if (!g1 || !g2) {
        fprintf(stderr, "Failed to create graphs\n");
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }
    
    // Load first graph
    print_memory_usage(g1, g2, vmap, "initial creation");
    int result = load_graph_sparse(g1_filename, g1, vmap);
    if (result != SUCCESS) {
        fprintf(stderr, "Failed to load first graph: error %d\n", result);
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }
    
    print_memory_usage(g1, g2, vmap, "loading first graph");
    
    // Load second graph
    result = load_graph_sparse(g2_filename, g2, vmap);
    if (result != SUCCESS) {
        fprintf(stderr, "Failed to load second graph: error %d\n", result);
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }
    
    print_memory_usage(g1, g2, vmap, "loading second graph");
    
    // Create and initialize matching array
    print_timestamp("Allocating matching array");
    int* matching = calloc(vmap->count, sizeof(int));
    if (!matching) {
        fprintf(stderr, "Failed to allocate matching array\n");
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }
    
    // Load matching
    result = load_matching(matching_filename, vmap, matching, vmap->count);
    if (result != SUCCESS) {
        fprintf(stderr, "Failed to load matching: error %d\n", result);
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        free(matching);
        return EXIT_FAILURE;
    }
    
    print_timestamp("Starting score calculation");
    fprintf(stderr, "Processing graphs with:\n");
    fprintf(stderr, "  Graph 1: %zu vertices and %zu edges\n", g1->num_vertices, g1->num_edges);
    fprintf(stderr, "  Graph 2: %zu vertices and %zu edges\n", g2->num_vertices, g2->num_edges);
    
    // Calculate and print score
    int score = calculate_score_sparse(g1, g2, matching);
    if (score >= 0) {
        print_timestamp("Calculation complete");
        printf("%d\n", score);
    } else {
        fprintf(stderr, "Error calculating score\n");
        free_vertex_map(vmap);
        free_graph(g1);
        free_graph(g2);
        free(matching);
        return EXIT_FAILURE;
    }
    
    // Cleanup
    print_timestamp("Cleaning up");
    free_vertex_map(vmap);
    free_graph(g1);
    free_graph(g2);
    free(matching);
    
    print_timestamp("Program completed successfully");
    return EXIT_SUCCESS;
}
