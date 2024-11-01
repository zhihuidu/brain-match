#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>  // Added for INT_MAX

#define MAX_VERTICES 20000
#define MAX_LINE_LENGTH 256
#define MAX_WEIGHT 1000  // Adjust based on expected maximum weight

typedef struct {
    char id[64];
    int degree;
    int total_weight;
} Vertex;

typedef struct {
    Vertex* vertices;
    int count;
} VertexSet;

typedef struct {
    int weight;
    int frequency;
} WeightFreq;

// Weight statistics structure
typedef struct {
    int* weights;
    int count;
    int min;
    int max;
    double mean;
    double median;
    double std_dev;
    // Histogram data
    WeightFreq* distribution;
    int unique_weights;
} WeightStats;

// Function to compare weights for sorting
int compare_weights(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

// Function to compare weight frequencies for sorting
int compare_weight_freq(const void* a, const void* b) {
    return ((WeightFreq*)b)->frequency - ((WeightFreq*)a)->frequency;
}

// Function to check if vertex exists and return its index
int find_vertex(VertexSet* set, const char* id) {
    for (int i = 0; i < set->count; i++) {
        if (strcmp(set->vertices[i].id, id) == 0) {
            return i;
        }
    }
    return -1;
}

// Function to add or update vertex
void add_vertex(VertexSet* set, const char* id, int weight) {
    int index = find_vertex(set, id);
    if (index == -1) {
        strcpy(set->vertices[set->count].id, id);
        set->vertices[set->count].degree = 1;
        set->vertices[set->count].total_weight = weight;
        set->count++;
    } else {
        set->vertices[index].degree++;
        set->vertices[index].total_weight += weight;
    }
}

void print_usage(const char* program_name) {
    printf("Usage: %s [-v] <csv_filename>\n", program_name);
    printf("Options:\n");
    printf("  -v    Enable verbose output\n");
    printf("  csv_filename    Path to the CSV file containing graph edges\n");
}

char get_vertex_type(const char* id) {
    return tolower(id[0]);
}

// Function to calculate weight statistics
WeightStats calculate_weight_stats(int* weights, int count) {
    WeightStats stats;
    stats.weights = weights;
    stats.count = count;
    
    // Sort weights for easier calculations
    qsort(weights, count, sizeof(int), compare_weights);
    
    // Calculate basic statistics
    stats.min = weights[0];
    stats.max = weights[count - 1];
    
    // Calculate mean
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += weights[i];
    }
    stats.mean = sum / count;
    
    // Calculate median
    if (count % 2 == 0) {
        stats.median = (weights[count/2 - 1] + weights[count/2]) / 2.0;
    } else {
        stats.median = weights[count/2];
    }
    
    // Calculate standard deviation
    double sum_squared_diff = 0;
    for (int i = 0; i < count; i++) {
        double diff = weights[i] - stats.mean;
        sum_squared_diff += diff * diff;
    }
    stats.std_dev = sqrt(sum_squared_diff / count);
    
    // Calculate weight distribution
    stats.distribution = malloc(MAX_WEIGHT * sizeof(WeightFreq));
    stats.unique_weights = 0;
    
    int current_weight = weights[0];
    int current_freq = 1;
    
    for (int i = 1; i < count; i++) {
        if (weights[i] == current_weight) {
            current_freq++;
        } else {
            stats.distribution[stats.unique_weights].weight = current_weight;
            stats.distribution[stats.unique_weights].frequency = current_freq;
            stats.unique_weights++;
            current_weight = weights[i];
            current_freq = 1;
        }
    }
    // Add last weight
    stats.distribution[stats.unique_weights].weight = current_weight;
    stats.distribution[stats.unique_weights].frequency = current_freq;
    stats.unique_weights++;
    
    // Sort distribution by frequency
    qsort(stats.distribution, stats.unique_weights, sizeof(WeightFreq), compare_weight_freq);
    
    return stats;
}

void print_weight_distribution(WeightStats* stats) {
    printf("\nWeight Distribution Analysis:\n");
    printf("---------------------------\n");
    printf("Basic Statistics:\n");
    printf("- Minimum weight: %d\n", stats->min);
    printf("- Maximum weight: %d\n", stats->max);
    printf("- Mean weight: %.2f\n", stats->mean);
    printf("- Median weight: %.2f\n", stats->median);
    printf("- Standard deviation: %.2f\n", stats->std_dev);
    
    printf("\nMost Common Weights (Top 10):\n");
    int top_n = (stats->unique_weights < 10) ? stats->unique_weights : 10;
    for (int i = 0; i < top_n; i++) {
        printf("- Weight %d: %d occurrences (%.1f%%)\n", 
               stats->distribution[i].weight,
               stats->distribution[i].frequency,
               (100.0 * stats->distribution[i].frequency) / stats->count);
    }
    
    // Print weight range distribution
    printf("\nWeight Range Distribution:\n");
    int ranges[6] = {0}; // 0-10, 11-50, 51-100, 101-500, 501-1000, >1000
    for (int i = 0; i < stats->count; i++) {
        if (stats->weights[i] <= 10) ranges[0]++;
        else if (stats->weights[i] <= 50) ranges[1]++;
        else if (stats->weights[i] <= 100) ranges[2]++;
        else if (stats->weights[i] <= 500) ranges[3]++;
        else if (stats->weights[i] <= 1000) ranges[4]++;
        else ranges[5]++;
    }
    
    const char* range_labels[] = {
        "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"
    };
    
    for (int i = 0; i < 6; i++) {
        printf("- %s: %d edges (%.1f%%)\n", 
               range_labels[i], 
               ranges[i], 
               (100.0 * ranges[i]) / stats->count);
    }
}

void print_verbose_stats(VertexSet* set, int* weights, int weight_count) {
    // Calculate vertex type statistics
    int type_counts[128] = {0};
    int max_degree = 0;
    int min_degree = INT_MAX;
    long total_weight = 0;
    
    for (int i = 0; i < set->count; i++) {
        char type = get_vertex_type(set->vertices[i].id);
        type_counts[(int)type]++;
        max_degree = (set->vertices[i].degree > max_degree) ? set->vertices[i].degree : max_degree;
        min_degree = (set->vertices[i].degree < min_degree) ? set->vertices[i].degree : min_degree;
        total_weight += set->vertices[i].total_weight;
    }

    printf("\nDetailed Statistics:\n");
    printf("-------------------\n");
    printf("Vertex Distribution:\n");
    for (int i = 0; i < 128; i++) {
        if (type_counts[i] > 0) {
            printf("- Vertices starting with '%c': %d\n", (char)i, type_counts[i]);
        }
    }
    
    printf("\nDegree Statistics:\n");
    printf("- Maximum vertex degree: %d\n", max_degree);
    printf("- Minimum vertex degree: %d\n", min_degree);
    printf("- Average vertex degree: %.2f\n", (float)total_weight / set->count);
    
    printf("\nTop 5 vertices by degree:\n");
    int* indices = malloc(set->count * sizeof(int));
    for (int i = 0; i < set->count; i++) indices[i] = i;
    
    for (int i = 0; i < 5 && i < set->count; i++) {
        for (int j = i + 1; j < set->count; j++) {
            if (set->vertices[indices[j]].degree > set->vertices[indices[i]].degree) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    for (int i = 0; i < 5 && i < set->count; i++) {
        printf("- %s: %d connections, total weight: %d\n",
               set->vertices[indices[i]].id,
               set->vertices[indices[i]].degree,
               set->vertices[indices[i]].total_weight);
    }
    
    free(indices);
    
    // Calculate and print weight distribution
    WeightStats weight_stats = calculate_weight_stats(weights, weight_count);
    print_weight_distribution(&weight_stats);
    
    free(weight_stats.distribution);
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    const char* filename = NULL;
    
    if (argc < 2 || argc > 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (argc == 2) {
        filename = argv[1];
    } else {
        if (strcmp(argv[1], "-v") == 0) {
            verbose = 1;
            filename = argv[2];
        } else if (strcmp(argv[2], "-v") == 0) {
            verbose = 1;
            filename = argv[1];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file '%s'\n", filename);
        return 1;
    }

    VertexSet set;
    set.vertices = (Vertex*)malloc(MAX_VERTICES * sizeof(Vertex));
    set.count = 0;

    // Array to store all weights for distribution analysis
    int* weights = malloc(MAX_VERTICES * MAX_VERTICES * sizeof(int)); // Upper bound
    int weight_count = 0;

    if (!set.vertices || !weights) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int edges = 0;
    
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char from_id[64], to_id[64];
        int weight;
        line_count++;
        
        if (sscanf(line, "%[^,],%[^,],%d", from_id, to_id, &weight) == 3) {
            add_vertex(&set, from_id, weight);
            add_vertex(&set, to_id, weight);
            weights[weight_count++] = weight;
            edges++;
            
            if (verbose && edges % 1000 == 0) {
                printf("Processed %d edges...\n", edges);
            }
        } else {
            printf("Warning: Malformed line %d in file %s\n", line_count, filename);
        }
    }

    printf("\nResults for %s:\n", filename);
    printf("Processed %d edges\n", edges);
    printf("Found %d unique vertices\n", set.count);

    if (verbose) {
        print_verbose_stats(&set, weights, weight_count);
    }

    free(set.vertices);
    free(weights);
    fclose(file);
    return 0;
}
