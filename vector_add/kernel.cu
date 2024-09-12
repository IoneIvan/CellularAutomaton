#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <stdint.h>
#define N 128 // Size of the grid
#define CELL_SIZE 5 // Size of each cell
#define WINDOW_SIZE (N * CELL_SIZE) // Size of the window
#define MAX_ENERGY 255 // Maximum energy for cells
#define INITIAL_ENERGY (MAX_ENERGY/10) // Initial energy for cells
#define GRAPH_WIDTH 256 // Width of the graph
#define GRAPH_HEIGHT (N*5) // Height of the graph
#define PIXEL_PER_UNIT 1 // Pixels per unit in the graph

struct Cell {
    uint16_t energy;
    uint8_t rotation;
};

__device__ Cell getCell(Cell* grid, int x, int y) {
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


__global__ void cellularAutomatonKernel(Cell* grid, Cell* nextGrid, unsigned long long seed, int taxes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize CURAND state
    curandState state;
    curand_init(seed, y * N + x, 0, &state); // Unique seed for each thread

    if (x < N && y < N) {
        Cell currentCell = getCell(grid, x, y);

        // Energy Loss
        if (currentCell.energy > 0) {
            currentCell.energy-=1 + currentCell.energy/ taxes;
        }

        // Energy Gain
        int facingNeighbors = countFacingNeighbors(grid, x, y, currentCell.rotation, taxes);
        currentCell.energy += facingNeighbors;

        // Random Rotation
        currentCell.rotation = curand(&state) % 9; // Random rotation between 0 and 8

        setCell(nextGrid, x, y, currentCell);
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
// Function to render the grid based on rotation
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

// Function to render the graph
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
        if (energy >= 0) {
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
            grid[y * N + x].energy = INITIAL_ENERGY;
            grid[y * N + x].rotation = rand() % 8; // Random rotation between 0 and 7
        }
    }
}
int main(int argc, char* argv[]) {
    Cell* grid;
    Cell* nextGrid;
    cudaMallocManaged(&grid, N * N * sizeof(Cell));
    cudaMallocManaged(&nextGrid, N * N * sizeof(Cell));

    int taxes = MAX_ENERGY; // Initialize taxes

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
                }
                if (event.key.keysym.sym == SDLK_a)
                {
                    isRender = !isRender;
                }
            }
        }

        ++frameCount;
        uint32_t seed = time(NULL) + frameCount;
        cellularAutomatonKernel << <numBlocks, threadsPerBlock >> > (grid, nextGrid, seed, taxes);
        cudaDeviceSynchronize();

        Cell* tmp = grid;
        grid = nextGrid;
        nextGrid = tmp;
        if (isRender)
        {
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