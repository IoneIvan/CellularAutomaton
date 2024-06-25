#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 64 // Size of the grid (NxN)
#define CELL_SIZE 10 // Size of each cell in pixels
#define WINDOW_SIZE (N * CELL_SIZE)
#define NUM_GENES 64

struct Cell {
    int alive; // Alive/Dead
    int energy; // Energy level
    int age; // Age
    int genes[NUM_GENES]; // Array of genes
    int currentActivity; // Current activity
};

__device__ int getCellAlive(Cell* grid, int x, int y) {
    return grid[y * N + x].alive;
}

__device__ void setCellAlive(Cell* grid, int x, int y, int value) {
    grid[y * N + x].alive = value;
}

__global__ void gameOfLifeKernel(Cell* currentGrid, Cell* nextGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        int liveNeighbors = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int neighborX = (x + i + N) % N;
                int neighborY = (y + j + N) % N;
                liveNeighbors += getCellAlive(currentGrid, neighborX, neighborY);
            }
        }

        Cell currentState = currentGrid[y * N + x];
        Cell nextState = currentState;
        if (currentState.alive == 1) {
            if (currentState.age > 100) nextState.alive = 0;
            if (liveNeighbors < 2 || liveNeighbors > 3) nextState.alive = 0;
            nextState.age++;
            nextState.energy--; // Example: reduce energy over time
        }
        else {
            if (liveNeighbors == 3) nextState.alive = 1;
            nextState.age = 0;
            nextState.energy = 100; // Example: reset energy on birth
        }
        nextGrid[y * N + x] = nextState;
    }
}

void renderGrid(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            if (grid[y * N + x].alive) {
                SDL_Rect cell;
                cell.x = x * CELL_SIZE;
                cell.y = y * CELL_SIZE;
                cell.w = CELL_SIZE;
                cell.h = CELL_SIZE;
                SDL_RenderFillRect(renderer, &cell);
            }
        }
    }

    SDL_RenderPresent(renderer);
}

int main(int argc, char* argv[]) {
    Cell* currentGrid;
    Cell* nextGrid;
    cudaMallocManaged(&currentGrid, N * N * sizeof(Cell));
    cudaMallocManaged(&nextGrid, N * N * sizeof(Cell));

    // Initialize the grid with random pattern
    srand(time(NULL));
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            currentGrid[y * N + x].alive = rand() % 2;
            currentGrid[y * N + x].energy = rand() % 100;
            currentGrid[y * N + x].age = rand() % 100;
            currentGrid[y * N + x].currentActivity = rand() % 10;
            for (int i = 0; i < NUM_GENES; ++i) {
                currentGrid[y * N + x].genes[i] = rand() % 2;
            }
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Game of Life",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_SIZE,
        WINDOW_SIZE,
        SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bool quit = false;
    SDL_Event event;

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
        }

        gameOfLifeKernel << <numBlocks, threadsPerBlock >> > (currentGrid, nextGrid);
        cudaDeviceSynchronize();

        Cell* temp = currentGrid;
        currentGrid = nextGrid;
        nextGrid = temp;

        renderGrid(renderer, currentGrid);
        //SDL_Delay(100); // Delay for 100ms
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cudaFree(currentGrid);
    cudaFree(nextGrid);

    return 0;
}
