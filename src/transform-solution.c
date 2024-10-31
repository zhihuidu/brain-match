#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1024
#define MAX_ID_LENGTH 32

void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s <input_solution_file.csv>\n", program_name);
    fprintf(stderr, "Converts solution file to matching format by adding 'm' and 'f' prefixes\n");
}

// Function to validate that string contains only digits
int is_valid_number(const char *str) {
    while (*str) {
        if (!isdigit(*str)) return 0;
        str++;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    FILE *input, *output;
    char line[MAX_LINE];
    char male_id[MAX_ID_LENGTH], female_id[MAX_ID_LENGTH];
    int line_number = 0;
    
    // Check command line arguments
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Open input file
    input = fopen(argv[1], "r");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", argv[1]);
        return 1;
    }

    // Create output filename by adding "_matching" before .csv
    char output_filename[MAX_LINE];
    char *dot = strrchr(argv[1], '.');
    if (dot) {
        strncpy(output_filename, argv[1], dot - argv[1]);
        output_filename[dot - argv[1]] = '\0';
        strcat(output_filename, "_matching.csv");
    } else {
        strcpy(output_filename, argv[1]);
        strcat(output_filename, "_matching.csv");
    }

    // Open output file
    output = fopen(output_filename, "w");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_filename);
        fclose(input);
        return 1;
    }

    // Write header for output file
    fprintf(output, "Male Node ID,Female Node ID\n");

    // Skip first line (header) of input file
    fgets(line, MAX_LINE, input);
    line_number++;

    // Process each data line
    while (fgets(line, MAX_LINE, input)) {
        line_number++;
        
        // Check for malformed lines
        if (sscanf(line, "%[^,],%[^\n\r]", male_id, female_id) != 2) {
            fprintf(stderr, "Error on line %d: Malformed input line\n", line_number);
            fprintf(stderr, "Line content: %s", line);
            fclose(input);
            fclose(output);
            return 1;
        }

        // Validate IDs contain only numbers
        if (!is_valid_number(male_id) || !is_valid_number(female_id)) {
            fprintf(stderr, "Error on line %d: IDs must contain only numbers\n", line_number);
            fprintf(stderr, "Male ID: %s, Female ID: %s\n", male_id, female_id);
            fclose(input);
            fclose(output);
            return 1;
        }

        // Write to output file with 'm' and 'f' prefixes
        fprintf(output, "m%s,f%s\n", male_id, female_id);
    }

    printf("Successfully processed %d lines\n", line_number);
    printf("Created %s\n", output_filename);

    fclose(input);
    fclose(output);
    return 0;
}
