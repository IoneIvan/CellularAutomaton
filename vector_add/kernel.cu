#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include <iostream>
#include <fstream>
#include <sys/stat.h> // For mkdir on POSIX systems
#include <filesystem> // For create_directory on C++17

#include <chrono>

using namespace std;


#define N 128 // Size of the grid (NxN)
#define CELL_SIZE 4 // Size of each cell in pixels
#define WINDOW_SIZE (N * CELL_SIZE)
#define MAX_ENERGY 1000
#define GENES_SIZE 5
#define COMMAND_NUMBER 20

struct Cell {
    int energy;
    int genes[GENES_SIZE];
    int activatedGene;
    int type;
    int rotation;
};

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int id = y * gridDim.x * blockDim.x + x;

    if (x < N && y < N) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__ float getRandomValue(curandState* globalState, int x, int y) {
    int id = y * gridDim.x * blockDim.x + x;
    curandState localState = globalState[id];
    float randomValue = curand_uniform(&localState);
    globalState[id] = localState;
    return randomValue;
}

__device__ Cell getCell(Cell* grid, int x, int y) {
    return grid[y * N + x];
}

__device__ void setCell(Cell* grid, int x, int y, Cell cell) {
    grid[y * N + x] = cell;
}


__device__ int getNeighborX(int x, int rotation) {
    // Define the direction based on the rotation (0-7)
    const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    return (x + dx[rotation] + N) % N;
}

__device__ int getNeighborY(int y, int rotation) {
    // Define the direction based on the rotation (0-7)
    const int dy[] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    return (y + dy[rotation]);
}

__device__ bool genesAreSimilar(Cell* cell1, Cell* cell2) {
    int differences = 0;

    // Compare genes
    for (int i = 0; i < GENES_SIZE; ++i) {
        if (cell1->genes[i] != cell2->genes[i]) {
            differences++;
            if (differences > 1) {
                return false; // More than one gene difference
            }
        }
    }

    return true; // At most one gene difference
}


__global__ void cellularAutomatonKernel(Cell* currentGrid, Cell* nextGrid, curandState* globalState) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        Cell cell = getCell(currentGrid, x, y);
        Cell nextCell = cell;
        if (cell.type == 1) {
            // Energy accumulation
            int command = cell.genes[cell.activatedGene];
            cell.activatedGene = (cell.activatedGene + 1) % GENES_SIZE;


            // Reproduction for cells of type 1
            if (cell.energy >= MAX_ENERGY) {
                int neighborX = getNeighborX(x, cell.rotation);
                int neighborY = getNeighborY(y, cell.rotation);

                if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                    Cell neighborCell = getCell(currentGrid, neighborX, neighborY);

                    if (neighborCell.type == 0) { // Empty cell found
                        cell.energy /= 2;
                        cell.rotation = (int)(getRandomValue(globalState, x, y) * 8); // Random rotation between 0 and 7
                        cell.rotation = (cell.rotation + 1 + 8) % 8; // Random rotation between 0 and 7
                    }
                    else {
                        cell.type = 0;
                    }
                }
            }

            int sharedEnergyAcumulated = 0;
            int numberSharedEnergy = 0;
            for (int i = 0; i < 8 && cell.type != 0; ++i) {
                int neighborX = getNeighborX(x, i);
                int neighborY = getNeighborY(y, i);

                if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                    Cell neighborCell = getCell(currentGrid, neighborX, neighborY);

                    if (neighborCell.type == 1) {
                        int command = cell.genes[cell.activatedGene];
                        cell.activatedGene = (cell.activatedGene + 1) % GENES_SIZE;
                        int targetX = 0;
                            int targetY = 0;
                        switch (command)
                        {
                        case 16:
                            cell.energy += neighborCell.energy / 9;
                            break;
                        case 17: // New attack command
                            targetX = getNeighborX(neighborX, neighborCell.rotation);
                            targetY = getNeighborY(neighborY, neighborCell.rotation);

                            if (targetX == x && targetY == y) { // Check if the target is the current cell

                                if (neighborCell.energy > 2 * cell.energy + 250) {
                                    cell.energy = 0;
                                    cell.type = 0;
                                }
                                else {
                                    cell.energy -= cell.energy / 4; // Lose 25% of energy
                                }
                            }
                        case 18:
                            targetX = getNeighborX(neighborX, neighborCell.rotation);
                            targetY = getNeighborY(neighborY, neighborCell.rotation);

                            if (targetX == x && targetY == y) { // Check if the target is the current cell
                                sharedEnergyAcumulated += neighborCell.energy;
                                ++numberSharedEnergy;
                            }
                            break;
                        }
                    }

                }
            }



            // Check energy depletion
            if (cell.energy <= 0 || cell.type == 0) {
                cell.type = 0;
                cell.energy = 0;
            }
            else
            {
                int neighborX = 0;
                int neighborY = 0;

                if (numberSharedEnergy > 0)
                {
                    cell.energy = (cell.energy + sharedEnergyAcumulated) / (numberSharedEnergy + 1);
                }

                switch (command)
                {
                case 8:
                    cell.energy += 25;
                    break;
                case 9:
                    // Change activatedGene based on neighbor's status
                    neighborX = getNeighborX(x, cell.rotation);
                    neighborY = getNeighborY(y, cell.rotation);

                    if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                        Cell neighborCell = getCell(currentGrid, neighborX, neighborY);
                        if (neighborCell.type == 1) {
                            if (genesAreSimilar(&cell, &neighborCell))
                            {
                                cell.activatedGene = cell.genes[(cell.activatedGene + 1) % GENES_SIZE]; // Move to next gene
                            }
                            else
                            {
                                cell.activatedGene = cell.genes[(cell.activatedGene + 2) % GENES_SIZE]; // Move to next gene
                            }
                        }
                        else {
                            cell.activatedGene = cell.genes[(cell.activatedGene + 3) % GENES_SIZE]; // Skip next gene (wrap around)
                        }
                    }
                    break;

                case 10:
                    // Change rotation based on gene value
                    int geneValue = cell.genes[cell.activatedGene];
                    cell.rotation = (cell.rotation + geneValue) % 8;
                    cell.activatedGene = (cell.activatedGene + 1) % GENES_SIZE;
                    break;
                case 11:
                    // Change rotation based on gene value
                    cell.rotation = (cell.genes[cell.activatedGene]) % 8;
                    cell.activatedGene = (cell.activatedGene + 1) % GENES_SIZE;
                    break;
                case 12:
                    // Change activatedGene based on neighbor's status
                    neighborX = getNeighborX(x, cell.rotation);
                    neighborY = getNeighborY(y, cell.rotation);

                    if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                        Cell neighborCell = getCell(currentGrid, neighborX, neighborY);
                        if (neighborCell.type == 1) {
                            if (neighborCell.energy > cell.genes[(cell.activatedGene) % GENES_SIZE] * MAX_ENERGY / COMMAND_NUMBER)
                            {
                                cell.activatedGene = cell.genes[(cell.activatedGene + 1) % GENES_SIZE]; // Move to next gene
                            }
                            else
                            {
                                cell.activatedGene = cell.genes[(cell.activatedGene + 2) % GENES_SIZE]; // Move to next gene
                            }
                        }
                        else {
                            cell.activatedGene = cell.genes[(cell.activatedGene + 3) % GENES_SIZE]; // Skip next gene (wrap around)
                        }
                    }

                    break;
                case 16:
                    cell.energy /= 9;
                    break;
                case 17: // New attack command
                    int targetX = getNeighborX(x, cell.rotation);
                    int targetY = getNeighborY(y, cell.rotation);

                    if (targetX >= 0 && targetX < N && targetY >= 0 && targetY < N) {
                        Cell targetCell = getCell(currentGrid, targetX, targetY);

                        if (targetCell.type == 1) { // Only attack active cells
                            if (cell.energy > 2 * targetCell.energy + 250) {
                                cell.energy += targetCell.energy / 4; // Consume 25% of the target's energy
                            }
                            else {
                                cell.energy -= cell.energy / 4; // Lose 25% of energy
                            }
                        }
                    }
                    break;
                case 18:
                    // Change activatedGene based on neighbor's status
                    neighborX = getNeighborX(x, cell.rotation);
                    neighborY = getNeighborY(y, cell.rotation);

                    if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                        Cell neighborCell = getCell(currentGrid, neighborX, neighborY);
                        if (neighborCell.type == 1) {
                            cell.energy = (cell.energy + neighborCell.energy) / 2;
                            cell.activatedGene = cell.genes[(cell.activatedGene) % GENES_SIZE]; // Skip next gene (wrap around)

                        }
                        else {
                            cell.activatedGene = cell.genes[(cell.activatedGene + 1) % GENES_SIZE]; // Skip next gene (wrap around)
                        }
                    }
                    break;
                }

                //age mutation
                if (getRandomValue(globalState, x, y) < 0.0001) { // chance to mutate a gene
                    cell.genes[(int)(getRandomValue(globalState, x, y) * GENES_SIZE)] = (int)(getRandomValue(globalState, x, y) * COMMAND_NUMBER);
                }
                cell.energy -= 1;
            }
        }
        else if (cell.type == 0) {

            // Reproduction for cells of type 0
            Cell potentialParents[8];
            Cell bestParent;
            bestParent.energy = 0;
            int parentCount = 0;

            for (int i = 0; i < 8; ++i) {
                int neighborX = getNeighborX(x, i);
                int neighborY = getNeighborY(y, i);

                if (neighborX >= 0 && neighborX < N && neighborY >= 0 && neighborY < N) {
                    Cell neighborCell = getCell(currentGrid, neighborX, neighborY);

                    if (neighborCell.type == 1 && neighborCell.energy >= MAX_ENERGY && neighborCell.rotation == (i + 4) % 8) {
                        potentialParents[parentCount++] = neighborCell;
                        if (bestParent.energy < neighborCell.energy)
                        {
                            bestParent = neighborCell;
                        }
                    }
                }
            }

            if (parentCount > 0) {
                Cell chosenParent = potentialParents[(int)(getRandomValue(globalState, x, y) * parentCount)];
                cell = bestParent;
                cell.energy /= 2;

                if (getRandomValue(globalState, x, y) < 0.15) { // chance to mutate a gene
                    cell.genes[(int)(getRandomValue(globalState, x, y) * GENES_SIZE)] = (int)(getRandomValue(globalState, x, y) * COMMAND_NUMBER);
                }
                setCell(nextGrid, x, y, cell);
            }
        }

        nextCell = cell;
        setCell(nextGrid, x, y, nextCell);
    }

}
__device__ int countNeighbors(Cell* grid, int x, int y) {
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; // Skip the cell itself
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < N && ny >= 0 && ny < N && grid[ny * N + nx].type == 1) {
                ++count;
            }
        }
    }
    return count;
}

__global__ void cellularAutomatonGravityKernel(Cell* currentGrid, Cell* nextGrid, curandState* globalState) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        int index = y * N + x;

        // Copy the current cell to the next grid
        nextGrid[index] = currentGrid[index];



        // Apply gravity if the cell has fewer than 5 neighbors
        if (y < N - 1 && currentGrid[index].type == 1 && currentGrid[(y + 1) * N + x].type == 0) {
            // Count the neighbors
            int neighborCount = countNeighbors(currentGrid, x, y);
            if (neighborCount < 3)
            {
                //nextGrid[index].type = 0;                       // Current cell becomes empty
            }
        }
        else if (currentGrid[index].type == 0)
        {
            int neighborCount = countNeighbors(currentGrid, x, y - 1);
            if (neighborCount < 3)
            {
                //nextGrid[index] = currentGrid[(y - 1) * N + x]; // Move the cell down
            }
        }
    }
}



bool showRed = true;
bool showGreen = true;
bool showBlue = true;

void renderGrid(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {



            int red = 0;
            int green = 0;
            int blue = 0;

            if (showRed)
            {
                red = (grid[y * N + x].genes[GENES_SIZE - 1] * 255) / COMMAND_NUMBER;
                green = (grid[y * N + x].genes[GENES_SIZE - 2] * 255) / COMMAND_NUMBER;
                blue = (grid[y * N + x].genes[GENES_SIZE - 3] * 255) / COMMAND_NUMBER;
            }
            if (showGreen)
            {
                // Calculate the green component based on the cell's energy level
                green = (grid[y * N + x].energy * 255) / MAX_ENERGY;
                green = green < 0 ? 0 : green > 255 ? 255 : green; // Clamp between 0 and 255           
            }
            if (showBlue)
            {

                blue = grid[y * N + x].type ? 255 : 0; // Set blue to 255 if alive, 0 if dead
            }

            SDL_SetRenderDrawColor(renderer, red, green, blue, 255);
            SDL_Rect cell;
            cell.x = x * CELL_SIZE;
            cell.y = y * CELL_SIZE;
            cell.w = CELL_SIZE;
            cell.h = CELL_SIZE;
            SDL_RenderFillRect(renderer, &cell);


        }
    }

    SDL_RenderPresent(renderer);
}
void renderGridEnergy(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    int energy = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
                // Calculate the green component based on the cell's energy level
                energy = (grid[y * N + x].energy * 255) / MAX_ENERGY;
                energy = energy < 0 ? 0 : energy > 255 ? 255 : energy; // Clamp between 0 and 255           

                energy = grid[y * N + x].type ? energy : 0; // Set blue to 255 if alive, 0 if dead
            

            SDL_SetRenderDrawColor(renderer, 0, energy, 0, 255);
            SDL_Rect cell;
            cell.x = x * CELL_SIZE;
            cell.y = y * CELL_SIZE;
            cell.w = CELL_SIZE;
            cell.h = CELL_SIZE;
            SDL_RenderFillRect(renderer, &cell);
        }
    }

    SDL_RenderPresent(renderer);
}
void renderGridFamilies(SDL_Renderer* renderer, Cell* grid) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {

            int red = 0;
            int green = 0;
            int blue = 0;
            red = (grid[y * N + x].genes[GENES_SIZE - 1] * 255) / COMMAND_NUMBER;
            green = (grid[y * N + x].genes[GENES_SIZE - 2] * 255) / COMMAND_NUMBER;
            blue = (grid[y * N + x].genes[GENES_SIZE - 3] * 255) / COMMAND_NUMBER;
            
            SDL_SetRenderDrawColor(renderer, red, green, blue, 255);
            SDL_Rect cell;
            cell.x = x * CELL_SIZE;
            cell.y = y * CELL_SIZE;
            cell.w = CELL_SIZE;
            cell.h = CELL_SIZE;
            SDL_RenderFillRect(renderer, &cell);
        }
    }
    SDL_RenderPresent(renderer);
}


void saveGridImage(SDL_Renderer* renderer, const char* path, int nameID) {
    char filename[256];  // Assuming a reasonable max path length

    // Create filename with path and imageNumber
    sprintf(filename, "%s/%d.bmp", path, nameID);

    SDL_Surface* surface = SDL_CreateRGBSurface(0, N * CELL_SIZE, N * CELL_SIZE, 32, 0, 0, 0, 0);
    if (surface) {
        SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch);
        SDL_SaveBMP(surface, filename);  // Save as BMP
        SDL_FreeSurface(surface);
    }
    else {
        fprintf(stderr, "Failed to create surface: %s\n", SDL_GetError());
    }
}



int main(int argc, char* argv[]) {
    Cell* currentGrid;
    Cell* nextGrid;
    curandState* devStates;
    cudaMallocManaged(&currentGrid, N * N * sizeof(Cell));
    cudaMallocManaged(&nextGrid, N * N * sizeof(Cell));
    cudaMalloc(&devStates, N * N * sizeof(curandState));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    setup_kernel << <numBlocks, threadsPerBlock >> > (devStates, time(NULL));
    cudaDeviceSynchronize();

    // Initialize the grid with a random pattern
    srand(time(NULL));

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            // Initialize cells with default values
            currentGrid[y * N + x] = { 0, {0}, 0, !(rand() % 25), 0 };

            if (currentGrid[y * N + x].type == 1) {
                for (int i = 0; i < GENES_SIZE; ++i) {
                    currentGrid[y * N + x].genes[i] = rand() % COMMAND_NUMBER;
                }
                currentGrid[y * N + x].energy = 500;
                currentGrid[y * N + x].rotation = rand() % 8;
            }
        }
    }

    // Set the middle cell with genes all set to 25
    int midX = N / 2;
    int midY = N / 2;
    Cell middleCell = { MAX_ENERGY, {25}, 0, 1, 0 };
    for (int i = 0; i < GENES_SIZE; ++i) {
        middleCell.genes[i] = 8;
    }
    currentGrid[midY * N + midX] = middleCell;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Cellular Automaton",
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


    string saveImages;
    cout << "Do you want to save the images? (yes/no): ";
    cin >> saveImages;

        string path;
    if (saveImages == "y") {
        cout << "Enter the path where you want to save the images: ";
        cin >> path;

        // Check if the directory exists
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            // Directory does not exist. Create it.
            cout << "Creating directory: " << path << endl;

            // CUDA-aware directory creation using CUDA API
            cudaError_t cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << endl;
                return 1;
            }

           

        }
        
        cout << "Images will be saved in: " << path << endl;
        // Now you can save your images in 'path'
    }
    else {
        cout << "Images will not be saved." << endl;
    }

    bool quit = false;
    SDL_Event event;
    int count = 0;
    int bmpCount = 0;
    int skipFrames = 10;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
            {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN)
            { // Add this block to handle key presses
                switch (event.key.keysym.sym) {
                case SDLK_r:
                    showRed = !showRed;
                    break;
                case SDLK_g:
                    showGreen = !showGreen;
                    break;
                case SDLK_b:
                    showBlue = !showBlue;
                    break;
                case SDLK_e:
                    ++skipFrames;
                    break;
                case SDLK_w:
                    if (skipFrames > 1)
                        --skipFrames;
                    break;
                case SDLK_q:
                    skipFrames = 10;
                    break;
                }
            }
        }
        auto start = std::chrono::high_resolution_clock::now();
        cellularAutomatonKernel << <numBlocks, threadsPerBlock >> > (currentGrid, nextGrid, devStates);
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "" << duration.count() << " microseconds" << std::endl;



        Cell* temp = currentGrid;
        currentGrid = nextGrid;
        nextGrid = temp;

     /*   cellularAutomatonGravityKernel << <numBlocks, threadsPerBlock >> > (currentGrid, nextGrid, devStates);
        cudaDeviceSynchronize();

        temp = currentGrid;
        currentGrid = nextGrid;
        nextGrid = temp;*/

        
        ++bmpCount;
            static string energyPath = path + "\\energy";
            static string familiesPath = path + "\\families";
        if (!(bmpCount % 200))
        {

            renderGridEnergy(renderer, currentGrid);

            saveGridImage(renderer, energyPath.c_str(), bmpCount);

            renderGridFamilies(renderer, currentGrid);

            saveGridImage(renderer, familiesPath.c_str(), bmpCount);
            //renderGrid(renderer, currentGrid);

            // Save rendered grid as an image

        }
        if (bmpCount % 10000 < 20)
        {
            renderGridEnergy(renderer, currentGrid);
            saveGridImage(renderer, energyPath.c_str(), -bmpCount);
        }

        ++count;
        if (!(count % skipFrames))
        {

            //renderGrid(renderer, currentGrid);
            // Save rendered grid as an image
            count = 0;
        }
        
    }
    cudaDeviceSynchronize();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cudaFree(currentGrid);
    cudaFree(nextGrid);
    cudaFree(devStates);

    return 0;
}
