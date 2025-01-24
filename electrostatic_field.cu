#include <cuda_runtime.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <chrono>  // For FPS calculation
#include <cstdlib>
#include <cstring>
#include <unistd.h>  // For getopt

#define Ck 1000000000.0f  // Coulomb's constant

#define CUDA_CHECK(cudaStatus)                                                 \
  if (cudaStatus != cudaSuccess)                                               \
    std::cout << cudaGetErrorString(cudaStatus) << std::endl;


struct Particle {
    float x, y;      // Position
    float vx, vy;    // Velocity
    int charge;      // +1 for proton, -1 for electron
};

struct ParticleSoA {
    float* x;
    float* y;
    float* vx;
    float* vy;
    int* charge;
};

__global__ void calculateElectrostaticField(ParticleSoA particles, int numParticles, int* field, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    float fieldStrength = 0.0f;

    for (int i = 0; i < numParticles; ++i) {
        float dx = particles.x[i] - idx;
        float dy = particles.y[i] - idy;
        float distanceSquared = dx * dx + dy * dy;
        // if (distanceSquared > 0.1f && distanceSquared < 10.0f) {
        if (distanceSquared > 0.1f ) {
            float force = Ck * particles.charge[i] / distanceSquared;
            fieldStrength += force;  // Accumulate field strength, positive for protons, negative for electrons
        }
    }
    // printf("[%d %d] %lf\n",idx, idy, fieldStrength);

    // Store the field strength as a signed integer
    int fieldValue = static_cast<int>(fieldStrength);
    field[idy * width + idx] = fieldValue;
}


__global__ void updateParticlePositions(ParticleSoA particles, int* field, int numParticles, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float fx = 0.0f, fy = 0.0f;

    for (int i = 0; i < numParticles; ++i) {
        if (i == idx) continue;

        float dx = particles.x[i] - particles.x[idx];
        float dy = particles.y[i] - particles.y[idx];
        float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 1e-6f) distanceSquared = 1e-6f; // Avoid division by zero

        // Determine the direction of the force based on charge polarity
        float chargeProduct = particles.charge[idx] * particles.charge[i];
        float forceMagnitude = fabsf(chargeProduct) / distanceSquared; // Use absolute value for magnitude
        float direction = (chargeProduct > 0) ? -1.0f : 1.0f; // Repel if charges are the same, attract if different

        distanceSquared = sqrtf(distanceSquared);

        fx += direction * forceMagnitude * dx / distanceSquared;
        fy += direction * forceMagnitude * dy / distanceSquared;
    }

    particles.vx[idx] += fx;
    particles.vy[idx] += fy;
    particles.x[idx] += particles.vx[idx];
    particles.y[idx] += particles.vy[idx];

    // Boundary conditions (bounce off the window edges)
    if (particles.x[idx] < 0 || particles.x[idx] > width) particles.vx[idx] = -particles.vx[idx];
    if (particles.y[idx] < 0 || particles.y[idx] > height) particles.vy[idx] = -particles.vy[idx];
}

// Function to parse command-line arguments
int parseArguments(int argc, char* argv[]) {
    int numParticles = 1000;  // Default value for number of particles

    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':  // -n option for number of particles
                numParticles = std::stoi(optarg);  // Convert the argument to integer
                if (numParticles <= 0) {
                    std::cerr << "Error: Number of particles must be a positive integer." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-n number_of_particles]" << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return numParticles;
}

int main(int argc, char* argv[]) {
    // Initialize SDL and create a window
    SDL_Init(SDL_INIT_VIDEO);

    int width = 600, height = 600;
    SDL_Window* window = SDL_CreateWindow("Electrostatic Field Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);



    int numParticles = parseArguments(argc, argv);

    ParticleSoA particlesSoA = {nullptr, nullptr, nullptr, nullptr, nullptr};
    particlesSoA.x = new float[numParticles];
    particlesSoA.y = new float[numParticles];
    particlesSoA.vx = new float[numParticles];
    particlesSoA.vy = new float[numParticles];
    particlesSoA.charge = new int[numParticles];

    // Initialize particles with random positions and velocities
    for (int i = 0; i < numParticles; ++i) {
        particlesSoA.x[i] = static_cast<float>(rand() % width); 
        particlesSoA.y[i] = static_cast<float>(rand() % height);
        particlesSoA.vx[i] = static_cast<float>(rand() % 200 - 100) / 100.0f;
        particlesSoA.vy[i] = static_cast<float>(rand() % 200 - 100) / 100.0f;
        particlesSoA.charge[i] = (rand() % 2) * 2 - 1;  // Randomly assign +1 or -1 charge
    }

    ParticleSoA d_particles;
    int* d_field;
    cudaMalloc(&d_particles.x, numParticles * sizeof(float));
    cudaMalloc(&d_particles.y, numParticles * sizeof(float));
    cudaMalloc(&d_particles.vx, numParticles * sizeof(float));
    cudaMalloc(&d_particles.vy, numParticles * sizeof(float));
    cudaMalloc(&d_particles.charge, numParticles * sizeof(int));
    cudaMalloc(&d_field, width * height * sizeof(int));
    cudaMemcpy(d_particles.x, particlesSoA.x, numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.y, particlesSoA.y, numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vx, particlesSoA.vx, numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vy, particlesSoA.vy, numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.charge, particlesSoA.charge, numParticles*sizeof(int), cudaMemcpyHostToDevice);

    // FPS calculation variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float fps = 0.0f;

    // Main loop
    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
        }

        {
            dim3 blockSize(32, 32);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
            calculateElectrostaticField<<<gridSize, blockSize>>>(d_particles, numParticles, d_field, width, height);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // int numThreads = 1024;
        // int numBlocks = (numParticles + numThreads - 1) / numThreads;
        int numSMs = 20; 
        int blocksPerSM = 2;
        int numBlocks = numSMs * blocksPerSM;

        int numThreads = 1024; 
        int numThreadsPerBlock = numThreads;

        int gridSize = (numParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;  // Adjust based on the problem size

        gridSize = max(gridSize, numBlocks);
        updateParticlePositions<<<gridSize, numThreadsPerBlock>>>(d_particles, d_field, numParticles, width, height);
        cudaDeviceSynchronize();

        std::vector<Particle> h_particles(numParticles);
        cudaMemcpy(particlesSoA.x, d_particles.x, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(particlesSoA.y, d_particles.y, numParticles * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<int> field(width * height);
        cudaMemcpy(field.data(), d_field, width * height * sizeof(int), cudaMemcpyDeviceToHost);

        SDL_RenderClear(renderer);  // Clear screen to white
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);  // Set background to white
        SDL_RenderFillRect(renderer, NULL);  // Fill the screen with white

        std::vector<SDL_Point> positivePoints;
        std::vector<SDL_Point> negativePoints;
        std::vector<SDL_Point> neutralPoints;

        positivePoints.reserve(width * height);
        negativePoints.reserve(width * height);
        neutralPoints.reserve(width * height);

        for (int i = 0; i < width * height; ++i) {
            int x = i % width;
            int y = i / width;
            int intensity = field[i];

            if (intensity > 0) {
                positivePoints.push_back({x, y});
            } else if (intensity == 0) {
                neutralPoints.push_back({x, y});
            } else {
                negativePoints.push_back({x, y});
            }
        }

        // Render positive points (red)
        // SDL_SetRenderDrawColor(renderer, 255, 0, 0, 155);  // Red
        // SDL_RenderDrawPoints(renderer, positivePoints.data(), positivePoints.size());

        // // Render neutral points (white)
        // SDL_SetRenderDrawColor(renderer, 255, 255, 255, 155);  // White
        // SDL_RenderDrawPoints(renderer, neutralPoints.data(), neutralPoints.size());

        // // Render negative points (blue)
        // SDL_SetRenderDrawColor(renderer, 0, 0, 255, 155);  // Blue
        // SDL_RenderDrawPoints(renderer, negativePoints.data(), negativePoints.size());


        // Batch all particle positions into a vector of SDL_Point
        std::vector<SDL_Point> particlePoints(numParticles);

        for (int i = 0; i < numParticles; ++i) {
            particlePoints[i] = {static_cast<int>(particlesSoA.x[i]), static_cast<int>(particlesSoA.y[i])};
        }

        // Render all particles as black dots in a single call
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Black color for particles
        SDL_RenderDrawPoints(renderer, particlePoints.data(), particlePoints.size());


        SDL_RenderPresent(renderer);
        // SDL_Delay(10);  // Delay to slow down simulation

        // FPS calculation
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = currentTime - lastTime;
        if (duration.count() >= 1.0f) {
            fps = frameCount / duration.count();
            frameCount = 0;
            lastTime = currentTime;

            // Update window title with FPS
            std::string title = "Electrostatic Field Simulation - FPS: " + std::to_string(fps);
            SDL_SetWindowTitle(window, title.c_str());
        }
    }

    // Clean up
    cudaFree(d_particles.x);
    cudaFree(d_particles.y);
    cudaFree(d_particles.vx);
    cudaFree(d_particles.vy);
    cudaFree(d_particles.charge);
    cudaFree(d_field);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
