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
#define INITIAL_ENERGY (200) // Initial energy for cells
#define GRAPH_WIDTH 256 // Width of the graph
#define GRAPH_HEIGHT (N*5) // Height of the graph
#define PIXEL_PER_UNIT 1 // Pixels per unit in the graph

#define INPUTS_COUNT 4
#define OUTPUT_COUNT 3

struct Cell {
    uint16_t energy;
    float neuronLayer[INPUTS_COUNT * OUTPUT_COUNT * OUTPUT_COUNT];
    uint8_t output[OUTPUT_COUNT];
};
struct Coordinates {
    int x;
    int y;
};
struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};
__device__ Cell getCell(Cell* grid, int x, int y) {
    return grid[y * N + x];
}
__device__ Cell getCell(Cell* grid, int x, int y, int r) {

    if (r == 0) ++x;
    if (r == 1) --x;
    if (r == 2) ++y;
    if (r == 3) --y;

    return grid[y * N + x];
}

__device__ Pixel getOutput(Pixel* grid, int x, int y) {
    return grid[y * N + x];
}

__device__ void setCell(Cell* grid, int x, int y, Cell value) {
    grid[y * N + x] = value;
}

__device__ uint8_t poorNeighbor(Cell* grid, int x, int y)
{
    uint8_t poorCells = 0;
    Coordinates result = { -1, -1 }; // Initialize with invalid coordinates
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {

            if (i == 0 && j == 0) continue; // Skip the current cell

            int neighborX = (x + i + N) % N; // Wrap around
            int neighborY = (y + j + N) % N; // Wrap around

            Cell neighbor = getCell(grid, neighborX, neighborY);
            Cell cell = getCell(grid, x, y);

            if (neighbor.energy < cell.energy)
            {
                ++poorCells;
            }
        }
    }

    return poorCells; // Return the coordinates
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


            if (neighbor.energy > cell.energy)
            {
                if (neighbor.energy == maxEnergy)
                {
                    if (rand % 8 == 0)
                    {
                        maxEnergy = neighbor.energy;
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

__global__ void cellularAutomatonKernel(Pixel* correct_output, Cell* grid, Cell* nextGrid, unsigned long long seed, int taxes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize CURAND state
    curandState state;
    curand_init(seed, y * N + x, 0, &state); // Unique seed for each thread

    if (x < N && y < N) {
        Cell cell = getCell(grid, x, y);

        //get populated
        Coordinates repCord = reproduceNeighbor(grid, x, y, curand(&state));
        if (repCord.x != -2)
        {
            cell = getCell(grid, repCord.x, repCord.y);
            if (curand(&state) % 4 == 0)
            {
                float randomValue = curand_uniform(&state);
                cell.neuronLayer[curand(&state) % (INPUTS_COUNT * OUTPUT_COUNT * OUTPUT_COUNT)] = randomValue - 0.5;
            }
            cell.energy /= 2;
        }
      

        // stop doing anything when dead
        if (cell.energy == 0)
        {
            setCell(nextGrid, x, y, cell);
            return;
        }

        //divide energy by the amout of cells that has energy less the half of this cell energy.
        cell.energy /= 2;

        
        
        //calculate output
        for (int i = 0; i < OUTPUT_COUNT; ++i) {
            cell.output[i] = 0;
            for (int j = 0; j < INPUTS_COUNT; ++j) {
                for (int k = 0; k < OUTPUT_COUNT; ++k) {
                    // Calculate the 1D index
                    int index = i * INPUTS_COUNT * OUTPUT_COUNT + j * OUTPUT_COUNT + k;
                    cell.output[i] += cell.neuronLayer[index] * getCell(grid, x, y, j).output[k];
                }
            }
        }


        //update energy
        Pixel outputCell = getOutput(correct_output, x, y);

        int error = abs(outputCell.r - cell.output[0]);
        error += abs(outputCell.g - cell.output[1]);
        error += abs(outputCell.b - cell.output[2]);

        cell.energy = (3*sizeof(uint8_t) - error);


        float spendEnergy = 0;
        // Initialize the array with some values (for demonstration)
        for (int i = 0; i < OUTPUT_COUNT; ++i) {
            for (int j = 0; j < INPUTS_COUNT; ++j) {
                for (int k = 0; k < OUTPUT_COUNT; ++k) {
                    // Calculate the 1D index
                    int index = i * INPUTS_COUNT * OUTPUT_COUNT + j * OUTPUT_COUNT + k;
                    spendEnergy += abs(cell.neuronLayer[index]);
                }
            }
        }
        //spendEnergy = 10;
        //cell.energy = cell.energy > 1 + spendEnergy ? cell.energy - 1 - spendEnergy : 0;

        setCell(nextGrid, x, y, cell);
    }
}

void renderGridCellOutput(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Cell cell = grid[y * N + x];

            SDL_SetRenderDrawColor(renderer, cell.output[0], cell.output[1], cell.output[2], 255); // Green for energy
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
void renderGridOutput(SDL_Renderer* renderer, Pixel* output) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Pixel outpt = output[y * N + x];


            SDL_SetRenderDrawColor(renderer, outpt.r, outpt.g, outpt.b, 255); // Green for energy
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


void resetCells(Cell* grid)
{
    // Initialize the grid with initial energy and random rotation
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            // Randomly assign energy
            if (rand() % 5 == 0)
                grid[y * N + x].energy = static_cast<float>(rand()) / RAND_MAX * INITIAL_ENERGY; // Random float
            else
                grid[y * N + x].energy = 0.0f;

            // Calculate output
            for (int i = 0; i < OUTPUT_COUNT; ++i) {
                for (int j = 0; j < INPUTS_COUNT; ++j) {
                    for (int k = 0; k < OUTPUT_COUNT; ++k) {
                        // Calculate the 1D index
                        int index = i * INPUTS_COUNT * OUTPUT_COUNT + j * OUTPUT_COUNT + k;
                        // Assign a random float value between -1.0 and 1.0 to neuronLayer
                        grid[y * N + x].neuronLayer[index] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random float between -1.0 and 1.0
                    }
                }
            }
            for (int i = 0; i < 3; ++i)
            {
                grid[y * N + x].output[i] = rand() % 256;
            }
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

void setOutput(Pixel* output) {
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

    // Extract the RGB channel values and store them in the output array
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            // Assuming imageData is in RGB format
            int index = (y * width + x) * 3; // Calculate the index for the RGB values
            output[y * N + x].r = imageData[index];        // Red channel
            output[y * N + x].g = imageData[index + 1];    // Green channel
            output[y * N + x].b = imageData[index + 2];    // Blue channel
        }
    }
}



int main(int argc, char* argv[]) {
    Pixel* correct_output;
    Cell* grid;
    Cell* nextGrid;
    cudaMallocManaged(&correct_output, N * N * sizeof(Pixel));
    cudaMallocManaged(&grid, N * N * sizeof(Cell));
    cudaMallocManaged(&nextGrid, N * N * sizeof(Cell));

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

   

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bool quit = false;
    SDL_Event event;

    unsigned long long seed = time(NULL); // Seed for random number generation
    uint32_t lastTime = SDL_GetTicks();
    uint32_t frameCount = 0;
    bool isRender = true;
    int menuType = 0;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_r) {
                    resetCells(grid);
                    setOutput(correct_output);
                }
                if (event.key.keysym.sym == SDLK_a)
                {
                    isRender = !isRender;
                }
                if (event.key.keysym.sym == SDLK_s)
                {
                    menuType = (++menuType)%2;
                }
            }
        }

        ++frameCount;
        uint32_t seed = time(NULL) + frameCount;
        cellularAutomatonKernel << <numBlocks, threadsPerBlock >> > (correct_output, grid, nextGrid, seed, 0);
        cudaDeviceSynchronize();

        Cell* tmp = grid;
        grid = nextGrid;
        nextGrid = tmp;
        if (isRender)
        {
            switch (menuType)
            {case 0:
                renderGridCellOutput(energyRenderer, grid);
                break;
            case 1:
                renderGridOutput(energyRenderer, correct_output);
                break;
            default:
                break;
            }
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
    SDL_Quit();

    cudaFree(grid);
    cudaFree(nextGrid);

    return 0;
}