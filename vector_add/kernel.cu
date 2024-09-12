#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <stdint.h>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sstream>

#define N 128 // Size of the grid
#define CELL_SIZE 5 // Size of each cell
#define WINDOW_SIZE (N * CELL_SIZE) // Size of the window
#define MAX_ENERGY 255 // Maximum energy for cells
#define INITIAL_ENERGY (MAX_ENERGY/10) // Initial energy for cells
#define GRAPH_WIDTH 256 // Width of the graph
#define GRAPH_HEIGHT (N*5) // Height of the graph
#define PIXEL_PER_UNIT 1 // Pixels per unit in the graph

#define GENES_COUNT 4
#define ACTIONS_COUT 16
struct Cell {
    uint16_t energy;
    uint8_t rotation;
    uint8_t genes[GENES_COUNT];
    uint8_t activeGene;
    uint8_t output;
};
struct Coordinates {
    int x;
    int y;
};
__device__ Cell getCell(Cell* grid, int x, int y) {
    return grid[y * N + x];
}
__device__ uint8_t getOutput(uint8_t* grid, int x, int y) {
    return grid[y * N + x];
}

__device__ void setCell(Cell* grid, int x, int y, Cell value) {
    grid[y * N + x] = value;
}

__device__ int countFacingNeighbors(Cell* grid, int x, int y, int rotation, int taxes) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue; // Skip the current cell
            int neighborX = (x + i + N) % N; // Wrap around
            int neighborY = (y + j + N) % N; // Wrap around
            Cell neighbor = getCell(grid, neighborX, neighborY);

            if (neighbor.energy == 0) continue;
            
            // Determine the rotation direction of the neighbor
            int neighborFacingDirection = neighbor.rotation % 8; // Neighbor's rotation facing direction

            // Check if the neighbor is facing the current cell
            if ((i == 0 && j == -1 && neighborFacingDirection == 0) || // Up
                (i == 1 && j == -1 && neighborFacingDirection == 1) || // Up-Right
                (i == 1 && j == 0 && neighborFacingDirection == 2) || // Right
                (i == 1 && j == 1 && neighborFacingDirection == 3) || // Down-Right
                (i == 0 && j == 1 && neighborFacingDirection == 4) || // Down
                (i == -1 && j == 1 && neighborFacingDirection == 5) || // Down-Left
                (i == -1 && j == 0 && neighborFacingDirection == 6) || // Left
                (i == -1 && j == -1 && neighborFacingDirection == 7)) { // Up-Left
                count += 1 + neighbor.energy / taxes;
            }
        }
    }
    return count;
}
__device__ int poorNeighbor(Cell* grid, int x, int y, float rand)
{
    uint16_t minEnergy = 0xFFFF;
    Coordinates result = { -1, -1 }; // Initialize with invalid coordinates
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue; // Skip the current cell
            int neighborX = (x + i + N) % N; // Wrap around
            int neighborY = (y + j + N) % N; // Wrap around
            Cell neighbor = getCell(grid, neighborX, neighborY);

            if (neighbor.energy == minEnergy)
            {
                if (rand < (1.0 / 8.0))
                {
                    result.x = neighborX;
                    result.y = neighborY;
                }
            }
            else if (neighbor.energy < minEnergy)
            {
                minEnergy = neighbor.energy;
                result.x = neighborX;
                result.y = neighborY;
            }
        }
    }

    return minEnergy; // Return the coordinates
}
__device__ Coordinates reproduceNeighbor(Cell* grid, int x, int y, int rand)
{
    uint16_t maxEnergy = 0x0000;
    Coordinates result = { -2, -2 }; // Initialize with invalid coordinates

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue; // Skip the current cell
            int neighborX = (x + i + N) % N; // Wrap around
            int neighborY = (y + j + N) % N; // Wrap around
            Cell neighbor = getCell(grid, neighborX, neighborY);
            Cell cell = getCell(grid, x, y);


            if (neighbor.energy >= MAX_ENERGY && neighbor.energy > 2 * cell.energy)
            {
                if (neighbor.energy == maxEnergy)
                {
                    if (rand % 8 == 0)
                    {
                        result.x = neighborX;
                        result.y = neighborY;
                    }
                }
                else if (neighbor.energy > maxEnergy)
                {
                    maxEnergy = neighbor.energy;
                    result.x = neighborX;
                    result.y = neighborY;
                }
            }
        }
    }

    return result; // Return the coordinates
}
__device__ Cell lookingAtNeighbor(Cell* grid, int x, int y)
{
   
    Cell cell = getCell(grid, x, y);

    // Determine the rotation direction of the neighbor
    int facingDirection = cell.rotation % 8; // Neighbor's rotation facing direction
    int nx = 0, ny = 0;
    switch (facingDirection)
    {
    case 0:
        nx = 0; ny = -1;
    case 1:
        nx = 1; ny = -1;
    case 2:
        nx = 1; ny = 0;
    case 3:
        nx = 1; ny = 1;
    case 4:
        nx = 0; ny = 1;
    case 5:
        nx = -1; ny = 1;
    case 6:
        nx = -1; ny = 0;
    case 7:
        nx = -1; ny = -1;
    }
    return getCell(grid, -nx, -ny);
}

__global__ void cellularAutomatonKernel(uint8_t* correct_output, Cell* grid, Cell* nextGrid, unsigned long long seed, int taxes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize CURAND state
    curandState state;
    curand_init(seed, y * N + x, 0, &state); // Unique seed for each thread

    if (x < N && y < N) {
        Cell cell = getCell(grid, x, y);

        if (cell.energy >= MAX_ENERGY)
        {
            if (poorNeighbor(grid, x, y, curand(&state)) * 2 > cell.energy)
            {
                cell.energy = 0;
            }
            else
            {
                cell.energy /= 2;
            }
        }
        Coordinates repCord = reproduceNeighbor(grid, x, y, curand(&state));
        if (repCord.x != -2)
        {
            cell = getCell(grid, repCord.x, repCord.y);
            if(curand(&state) % 4 == 0)
                cell.genes[curand(&state) % GENES_COUNT] = curand(&state) % ACTIONS_COUT;
            cell.energy /= 2;
        }

        // Energy Loss
        if (cell.energy > 0) {
            cell.energy -= 1 + cell.energy / taxes;
        }
        else
        {
            setCell(nextGrid, x, y, cell);
            return;
        }

        // Energy Gain
        int facingNeighbors = countFacingNeighbors(grid, x, y, cell.rotation, taxes);
        cell.energy += facingNeighbors;
        switch (cell.genes[(++cell.activeGene) % GENES_COUNT])
        {
        case 0:
            cell.rotation = curand(&state) % 9; // Random rotation between 0 and 8
            break;
        case 1:
            cell.rotation = (cell.rotation + cell.genes[(++cell.activeGene) % GENES_COUNT]) % 9; // Random rotation between 0 and 8
            break;
        
        case 2:
            cell.output = cell.genes[(++cell.activeGene) % GENES_COUNT];
            break;
        case 3:
            cell.output = (cell.output + cell.genes[(++cell.activeGene) % GENES_COUNT] - 128) % sizeof(cell.output);
            break;
        case 4:
            cell.output = (cell.output + lookingAtNeighbor(grid, x, y).output - 128) % sizeof(cell.output);
            break;
        case 5:
            cell.output = curand(&state) % 255;
            break;
        case 6:
            cell.output = 0;
        case 10:
            //cell.energy += 1;
            break;
        }
        uint8_t outputCell = getOutput(correct_output, x, y);
      
        cell.energy += (255 - abs(outputCell - cell.output))/51;
        

        setCell(nextGrid, x, y, cell);
    }
}

void renderGrid(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Cell cell = grid[y * N + x];


            if (cell.energy > 0) {

                Uint8 rotation = (8 - cell.rotation) * 32;
                Uint8 energy = cell.energy < MAX_ENERGY ? cell.energy * (255 / MAX_ENERGY) : 255;
                Uint8 superRich = cell.energy >= MAX_ENERGY ? (cell.energy - MAX_ENERGY) * (255 / MAX_ENERGY) : 0;
                Uint8 megaRch = 0;
                if (superRich > 255) 
                {
                    superRich = 255;
                    megaRch = 255;
                }
                SDL_SetRenderDrawColor(renderer, superRich, energy, megaRch, 255); // Green for energy
                SDL_Rect rect;
                rect.x = x * CELL_SIZE;
                rect.y = y * CELL_SIZE;
                rect.w = CELL_SIZE;
                rect.h = CELL_SIZE;
                SDL_RenderFillRect(renderer, &rect);
            }

        }
    }

    SDL_RenderPresent(renderer);
}
void renderGridOutput(SDL_Renderer* renderer, Cell* grid, uint8_t* output) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Cell cell = grid[y * N + x];


                SDL_SetRenderDrawColor(renderer, cell.output, cell.output, cell.output, 255); // Green for energy
                SDL_Rect rect;
                rect.x = x * CELL_SIZE;
                rect.y = y * CELL_SIZE;
                rect.w = CELL_SIZE;
                rect.h = CELL_SIZE;
                SDL_RenderFillRect(renderer, &rect);
            

        }
    }

    SDL_RenderPresent(renderer);
}

void renderRotationGrid(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Cell cell = grid[y * N + x];

            // Use rotation to determine color (for example, using rotation value)
            Uint8 rotationColor = (cell.rotation * 32) % 256; // Example color based on rotation
            SDL_SetRenderDrawColor(renderer, rotationColor, 0, 0, 255); // Blue for rotation
            SDL_Rect rect;
            rect.x = x;
            rect.y = y;
            rect.w = 1;
            rect.h = 1;
            SDL_RenderFillRect(renderer, &rect);
        }
    }

    SDL_RenderPresent(renderer);
}
void renderGraph(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
    SDL_RenderClear(renderer);

    // Draw axes
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black for axes
    SDL_RenderDrawLine(renderer, 50, GRAPH_HEIGHT, 50, 0); // Y-axis
    SDL_RenderDrawLine(renderer, 0, GRAPH_HEIGHT, GRAPH_WIDTH + 50, GRAPH_HEIGHT); // X-axis

    int energyDistribution[MAX_ENERGY * 8] = { 0 };

    // Calculate energy distribution
    for (int i = 0; i < N * N; ++i) {
        int energy = grid[i].energy;
        if (energy >= 0 && energy < MAX_ENERGY * 8) {
            energyDistribution[energy]++;
        }
    }

    // Draw bars for energy distribution
    for (int x = 0; x < MAX_ENERGY * 8; ++x) {
        int barHeight = energyDistribution[x]/2; // Height of the bar
        int y = GRAPH_HEIGHT - barHeight; // Y position for the top of the bar
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red for energy bars

        // Draw the bar as a rectangle
        SDL_Rect barRect;
        barRect.x = 50 + x; // X position
        barRect.y = y; // Y position
        barRect.w = 1; // Width of the bar
        barRect.h = barHeight; // Height of the bar
        SDL_RenderFillRect(renderer, &barRect); // Fill the rectangle to create the bar
    }

    SDL_RenderPresent(renderer);
}

void resetCells(Cell* grid)
{
    // Initialize the grid with initial energy and random rotation
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            //grid[y * N + x].energy = INITIAL_ENERGY;
            if (rand() % 5 == 0)
                grid[y * N + x].energy = rand() % INITIAL_ENERGY;
            else
                grid[y * N + x].energy = 0;
            grid[y * N + x].rotation = rand() % 8; // Random rotation between 0 and 7
            for (int i = 0; i < GENES_COUNT; ++i)
            {
                grid[y * N + x].genes[i] = rand() % ACTIONS_COUT;
            }
            grid[y * N + x].activeGene = 0;
        }
    }
}



// This function is a placeholder for loading the image
// In real scenarios, you would need a library like stb_image or OpenCV
bool loadImage(const std::string& path, std::vector<uint8_t>& imageData, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << path << std::endl;
        return false;
    }

    // Read the header
    std::string header;
    file >> header;
    if (header != "P6") {
        std::cerr << "Error: Only P6 PPM format is supported" << std::endl;
        return false;
    }

    // Read width, height, and maximum color value
    file >> width >> height;
    int maxColorValue;
    file >> maxColorValue;
    if (maxColorValue != 255) {
        std::cerr << "Error: Only 8-bit color PPM format is supported" << std::endl;
        return false;
    }

    // Skip the newline character after the header
    file.ignore(1);

    // Allocate memory for image data
    imageData.resize(width * height * 3);

    // Read pixel data
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    if (!file) {
        std::cerr << "Error: Unexpected end of file" << std::endl;
        return false;
    }

    return true;
}

void setOutput(uint8_t * output) {
    std::string path;
    std::cout << "Enter the path to the PNG or JPEG image: ";
    std::cin >> path;

    // Placeholder for image data
    std::vector<uint8_t> imageData;
    int width, height;

    // Load the image
    if (!loadImage(path, imageData, width, height)) {
        std::cerr << "Failed to load the image." << std::endl;
        return;
    }

    // Ensure the image fits within NxN
    if (width < N || height < N) {
        std::cerr << "Image is too small; it must be at least " << N << "x" << N << " pixels." << std::endl;
        return;
    }

    // Extract the red channel values and store them in the output array
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            // Assuming imageData is in RGB format
            output[y * N + x] = imageData[(y * width + x) * 3];  // Red channel
        }
    }
}



int main(int argc, char* argv[]) {
    uint8_t* correct_output;
    Cell* grid;
    Cell* nextGrid;
    cudaMallocManaged(&correct_output, N * N * sizeof(uint8_t));
    cudaMallocManaged(&grid, N * N * sizeof(Cell));
    cudaMallocManaged(&nextGrid, N * N * sizeof(Cell));

    int taxes = MAX_ENERGY; // Initialize taxes
    setOutput(correct_output);
    resetCells(grid);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    // Create the first window for energy
    SDL_Window* energyWindow = SDL_CreateWindow("Cell Energy",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_SIZE,
        WINDOW_SIZE,
        SDL_WINDOW_SHOWN);
    if (!energyWindow) {
        fprintf(stderr, "Could not create energy window: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* energyRenderer = SDL_CreateRenderer(energyWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!energyRenderer) {
        fprintf(stderr, "Could not create energy renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(energyWindow);
        SDL_Quit();
        return 1;
    }

    // Create the second window for rotation
    SDL_Window* rotationWindow = SDL_CreateWindow("Cell Rotation",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        N,
        N,
        SDL_WINDOW_SHOWN);
    if (!rotationWindow) {
        fprintf(stderr, "Could not create rotation window: %s\n", SDL_GetError());
        SDL_DestroyRenderer(energyRenderer);
        SDL_DestroyWindow(energyWindow);
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* rotationRenderer = SDL_CreateRenderer(rotationWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!rotationRenderer) {
        fprintf(stderr, "Could not create rotation renderer: %s\n", SDL_GetError());
        SDL_DestroyRenderer(energyRenderer);
        SDL_DestroyWindow(energyWindow);
        SDL_DestroyWindow(rotationWindow);
        SDL_Quit();
        return 1;
    }

    // Create the third window for the graph
    SDL_Window* graphWindow = SDL_CreateWindow("Energy Graph",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        GRAPH_WIDTH + 100, // Extra space for y-axis
        GRAPH_HEIGHT,
        SDL_WINDOW_SHOWN);
    if (!graphWindow) {
        fprintf(stderr, "Could not create graph window: %s\n", SDL_GetError());
        SDL_DestroyRenderer(energyRenderer);
        SDL_DestroyWindow(energyWindow);
        SDL_DestroyRenderer(rotationRenderer);
        SDL_DestroyWindow(rotationWindow);
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* graphRenderer = SDL_CreateRenderer(graphWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!graphRenderer) {
        fprintf(stderr, "Could not create graph renderer: %s\n", SDL_GetError());
        SDL_DestroyRenderer(energyRenderer);
        SDL_DestroyWindow(energyWindow);
        SDL_DestroyRenderer(rotationRenderer);
        SDL_DestroyWindow(rotationWindow);
        SDL_DestroyWindow(graphWindow);
        SDL_Quit();
        return 1;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bool quit = false;
    SDL_Event event;

    unsigned long long seed = time(NULL); // Seed for random number generation
    uint32_t lastTime = SDL_GetTicks();
    uint32_t frameCount = 0;
    bool isRender = true;
    bool showOutput = false;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_e) {
                    taxes++; // Increase taxes
                    printf("Current taxes: %d\n", taxes); // Print current taxes
                }
                else if (event.key.keysym.sym == SDLK_w) {
                   
                    if(taxes > 2)
                        taxes--; // Decrease taxes
                    printf("Current taxes: %d\n", taxes); // Print current taxes
                }
                else if (event.key.keysym.sym == SDLK_r) {
                    resetCells(grid);
                    setOutput(correct_output);
                }
                if (event.key.keysym.sym == SDLK_a)
                {
                    isRender = !isRender;
                }
                if (event.key.keysym.sym == SDLK_s)
                {
                    showOutput = !showOutput;
                }
            }
        }

        ++frameCount;
        uint32_t seed = time(NULL) + frameCount;
        cellularAutomatonKernel << <numBlocks, threadsPerBlock >> > (correct_output, grid, nextGrid, seed, taxes);
        cudaDeviceSynchronize();

        Cell* tmp = grid;
        grid = nextGrid;
        nextGrid = tmp;
        if (isRender)
        {
            if(showOutput)
                renderGridOutput(energyRenderer, grid, correct_output);
            else
                renderGrid(energyRenderer, grid);
            renderRotationGrid(rotationRenderer, grid);
            renderGraph(graphRenderer, grid); // Render the graph
        }
       

        // Calculate and print the frequency every second
        uint32_t currentTime = SDL_GetTicks();
        if (currentTime - lastTime >= 1000) { // If a second has passed
            printf("Loop executed %d times in the last second.\n", frameCount);
            frameCount = 0; // Reset frame count
            lastTime = currentTime; // Reset last time
        }

        // SDL_Delay(100); // Delay for 100ms
    }

    SDL_DestroyRenderer(energyRenderer);
    SDL_DestroyWindow(energyWindow);
    SDL_DestroyRenderer(rotationRenderer);
    SDL_DestroyWindow(rotationWindow);
    SDL_DestroyRenderer(graphRenderer);
    SDL_DestroyWindow(graphWindow);
    SDL_Quit();

    cudaFree(grid);
    cudaFree(nextGrid);

    return 0;
}