#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>  // Added for SIZE_MAX

#define INITIAL_HASH_SIZE 16384
#define INITIAL_BUFFER_SIZE 1024
#define RESIZE_FACTOR 2
#define MAX_ID_LENGTH 256

// Forward declarations
void cleanup_and_exit(const char* message);
static unsigned int hash(const char* str, size_t table_size);

typedef struct Vertex {
    char* id;
    int degree;
    long long total_weight;
    struct Vertex* next;
} Vertex;

typedef struct {
    Vertex** table;
    size_t size;
    size_t count;
    size_t collisions;
} VertexHashTable;

typedef struct {
    int weight;
    size_t frequency;
} WeightFreq;

typedef struct {
    long long min;
    long long max;
    double mean;
    double median;
    double std_dev;
    WeightFreq* distribution;
    size_t unique_weights;
    size_t total_count;
} WeightStats;

typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} WeightArray;

// Moved hash function definition before its first use
static unsigned int hash(const char* str, size_t table_size) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash & (table_size - 1);
}

// Error handling function
void cleanup_and_exit(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(1);
}

// Safe memory allocation wrapper
void* safe_malloc(size_t size, const char* error_msg) {
    void* ptr = malloc(size);
    if (!ptr) {
        cleanup_and_exit(error_msg);
    }
    return ptr;
}

// Safe memory reallocation wrapper
void* safe_realloc(void* ptr, size_t size, const char* error_msg) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        cleanup_and_exit(error_msg);
    }
    return new_ptr;
}

// Safe string duplication
char* safe_strdup(const char* str, const char* error_msg) {
    char* new_str = strdup(str);
    if (!new_str) {
        cleanup_and_exit(error_msg);
    }
    return new_str;
}

// Initialize weight array with error checking
WeightArray* create_weight_array() {
    WeightArray* wa = safe_malloc(sizeof(WeightArray), 
                                "Failed to allocate weight array structure");
    wa->capacity = INITIAL_BUFFER_SIZE;
    wa->size = 0;
    wa->data = safe_malloc(wa->capacity * sizeof(int), 
                          "Failed to allocate weight array data");
    return wa;
}

// Add weight to array with safety checks
void add_weight(WeightArray* wa, int weight) {
    if (wa->size >= wa->capacity) {
        // Check for overflow before multiplying
        if (wa->capacity > SIZE_MAX / RESIZE_FACTOR / sizeof(int)) {
            cleanup_and_exit("Weight array capacity overflow");
        }
        wa->capacity *= RESIZE_FACTOR;
        wa->data = safe_realloc(wa->data, wa->capacity * sizeof(int),
                               "Failed to resize weight array");
    }
    wa->data[wa->size++] = weight;
}

// Create hash table with error checking
VertexHashTable* create_hash_table() {
    VertexHashTable* ht = safe_malloc(sizeof(VertexHashTable),
                                    "Failed to allocate hash table structure");
    
    // Check for overflow before allocating table
    if (INITIAL_HASH_SIZE > SIZE_MAX / sizeof(Vertex*)) {
        cleanup_and_exit("Hash table size overflow");
    }
    
    ht->size = INITIAL_HASH_SIZE;
    ht->table = safe_malloc(ht->size * sizeof(Vertex*),
                           "Failed to allocate hash table array");
    memset(ht->table, 0, ht->size * sizeof(Vertex*));
    ht->count = 0;
    ht->collisions = 0;
    return ht;
}

// Resize hash table with safety checks
void resize_hash_table(VertexHashTable* ht) {
    // Check for overflow before calculating new size
    if (ht->size > SIZE_MAX / RESIZE_FACTOR / sizeof(Vertex*)) {
        cleanup_and_exit("Hash table resize overflow");
    }
    
    size_t new_size = ht->size * RESIZE_FACTOR;
    Vertex** new_table = safe_malloc(new_size * sizeof(Vertex*),
                                   "Failed to allocate resized hash table");
    memset(new_table, 0, new_size * sizeof(Vertex*));
    
    // Rehash all entries
    for (size_t i = 0; i < ht->size; i++) {
        Vertex* current = ht->table[i];
        while (current) {
            Vertex* next = current->next;
            unsigned int new_index = hash(current->id, new_size);
            current->next = new_table[new_index];
            new_table[new_index] = current;
            current = next;
        }
    }
    
    free(ht->table);
    ht->table = new_table;
    ht->size = new_size;
    ht->collisions = 0;
}

// Find or create vertex with safety checks
Vertex* find_or_create_vertex(VertexHashTable* ht, const char* id, int weight) {
    if (ht->collisions > (ht->size * 0.7)) {
        resize_hash_table(ht);
    }
    
    unsigned int index = hash(id, ht->size);
    Vertex* current = ht->table[index];
    
    while (current) {
        if (strcmp(current->id, id) == 0) {
            current->degree++;
            current->total_weight += weight;
            return current;
        }
        current = current->next;
        ht->collisions++;
    }
    
    // Create new vertex with safety checks
    Vertex* new_vertex = safe_malloc(sizeof(Vertex),
                                   "Failed to allocate new vertex");
    new_vertex->id = safe_strdup(id, "Failed to duplicate vertex ID");
    new_vertex->degree = 1;
    new_vertex->total_weight = weight;
    new_vertex->next = ht->table[index];
    ht->table[index] = new_vertex;
    ht->count++;
    
    return new_vertex;
}

// Calculate statistics with safety checks
WeightStats calculate_weight_stats(WeightArray* weights) {
    WeightStats stats;
    stats.total_count = weights->size;
    
    if (stats.total_count == 0) {
        memset(&stats, 0, sizeof(WeightStats));
        return stats;
    }
    
    stats.min = INT_MAX;
    stats.max = INT_MIN;
    double sum = 0;
    
    for (size_t i = 0; i < weights->size; i++) {
        int w = weights->data[i];
        if (w < stats.min) stats.min = w;
        if (w > stats.max) stats.max = w;
        sum += w;
    }
    stats.mean = sum / stats.total_count;
    
    // Check for integer overflow in range calculation
    if (stats.max > INT_MAX - stats.min) {
        cleanup_and_exit("Weight range overflow");
    }
    size_t range = stats.max - stats.min + 1;
    
    // Check for multiplication overflow
    if (range > SIZE_MAX / sizeof(size_t)) {
        cleanup_and_exit("Frequency array size overflow");
    }
    
    size_t* freq = safe_malloc(range * sizeof(size_t),
                              "Failed to allocate frequency array");
    memset(freq, 0, range * sizeof(size_t));
    
    double sum_squared_diff = 0;
    for (size_t i = 0; i < weights->size; i++) {
        int w = weights->data[i];
        freq[w - stats.min]++;
        double diff = w - stats.mean;
        sum_squared_diff += diff * diff;
    }
    stats.std_dev = sqrt(sum_squared_diff / stats.total_count);
    
    // Calculate median
    size_t median_pos = stats.total_count / 2;
    size_t count = 0;
    for (size_t i = 0; i < range; i++) {
        count += freq[i];
        if (count > median_pos) {
            stats.median = i + stats.min;
            break;
        }
    }
    
    // Count unique weights
    stats.unique_weights = 0;
    for (size_t i = 0; i < range; i++) {
        if (freq[i] > 0) stats.unique_weights++;
    }
    
    // Allocate and fill distribution array
    stats.distribution = safe_malloc(stats.unique_weights * sizeof(WeightFreq),
                                   "Failed to allocate weight distribution");
    size_t dist_index = 0;
    for (size_t i = 0; i < range; i++) {
        if (freq[i] > 0) {
            stats.distribution[dist_index].weight = i + stats.min;
            stats.distribution[dist_index].frequency = freq[i];
            dist_index++;
        }
    }
    
    free(freq);
    return stats;
}

void print_weight_distribution(WeightStats* stats, const int* weights, size_t weight_count) {
    printf("\nWeight Distribution Analysis:\n");
    printf("---------------------------\n");
    printf("Basic Statistics:\n");
    printf("- Minimum weight: %lld\n", stats->min);
    printf("- Maximum weight: %lld\n", stats->max);
    printf("- Mean weight: %.2f\n", stats->mean);
    printf("- Median weight: %.2f\n", stats->median);
    printf("- Standard deviation: %.2f\n", stats->std_dev);
    
    printf("\nMost Common Weights (Top 10):\n");
    int top_n = (stats->unique_weights < 10) ? stats->unique_weights : 10;
    for (int i = 0; i < top_n; i++) {
        printf("- Weight %d: %zu occurrences (%.1f%%)\n", 
               stats->distribution[i].weight,
               stats->distribution[i].frequency,
               (100.0 * stats->distribution[i].frequency) / stats->total_count);
    }
    
    // Print weight range distribution using raw weights
    printf("\nWeight Range Distribution:\n");
    size_t ranges[6] = {0}; // 0-10, 11-50, 51-100, 101-500, 501-1000, >1000
    
    for (size_t i = 0; i < weight_count; i++) {
        int w = weights[i];
        if (w <= 10) ranges[0]++;
        else if (w <= 50) ranges[1]++;
        else if (w <= 100) ranges[2]++;
        else if (w <= 500) ranges[3]++;
        else if (w <= 1000) ranges[4]++;
        else ranges[5]++;
    }
    
    const char* range_labels[] = {
        "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"
    };
    
    for (int i = 0; i < 6; i++) {
        printf("- %s: %zu edges (%.1f%%)\n", 
               range_labels[i], 
               ranges[i],
               (100.0 * ranges[i]) / weight_count);
    }
}

void process_graph(const char* filename, int verbose) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return;
    }
    
    VertexHashTable* ht = create_hash_table();
    WeightArray* weights = create_weight_array();
    
    char* buffer = NULL;
    size_t buffer_size = 0;
    size_t edges = 0;
    
    // Skip header
    if (getline(&buffer, &buffer_size, file) == -1) {
        cleanup_and_exit("Failed to read header line");
    }
    
    char* from_id = safe_malloc(MAX_ID_LENGTH, "Failed to allocate from_id buffer");
    char* to_id = safe_malloc(MAX_ID_LENGTH, "Failed to allocate to_id buffer");
    int weight;
    
    while (getline(&buffer, &buffer_size, file) != -1) {
        if (sscanf(buffer, "%[^,],%[^,],%d", from_id, to_id, &weight) == 3) {
            find_or_create_vertex(ht, from_id, weight);
            find_or_create_vertex(ht, to_id, weight);
            add_weight(weights, weight);
            edges++;
            
            if (verbose && edges % 1000000 == 0) {
                printf("Processed %zu million edges...\n", edges / 1000000);
            }
        }
    }
    
    printf("\nResults for %s:\n", filename);
    printf("Processed %zu edges\n", edges);
    printf("Found %zu unique vertices\n", ht->count);
    
    if (verbose) {
        WeightStats stats = calculate_weight_stats(weights);
        // Pass the raw weights array to print_weight_distribution
        print_weight_distribution(&stats, weights->data, weights->size);
        free(stats.distribution);
    }
    
    // Cleanup in reverse order of allocation
    for (size_t i = 0; i < ht->size; i++) {
        Vertex* current = ht->table[i];
        while (current) {
            Vertex* next = current->next;
            free(current->id);
            free(current);
            current = next;
        }
    }
    free(ht->table);
    free(ht);
    free(weights->data);
    free(weights);
    free(buffer);
    free(from_id);
    free(to_id);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s [-v] <csv_filename>\n", argv[0]);
        printf("Options:\n");
        printf("  -v    Enable verbose output\n");
        printf("  csv_filename    Path to the CSV file containing graph edges\n");
        return 1;
    }
    
    const char* filename;
    int verbose = 0;
    
    if (argc == 2) {
        filename = argv[1];
    } else {
        verbose = (strcmp(argv[1], "-v") == 0 || strcmp(argv[2], "-v") == 0);
        filename = (strcmp(argv[1], "-v") == 0) ? argv[2] : argv[1];
    }
    
    process_graph(filename, verbose);
    return 0;
}
