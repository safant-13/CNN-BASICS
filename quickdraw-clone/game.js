// Game logic and TensorFlow.js integration for Quick Draw Clone
class QuickDrawGame {
    constructor() {
        this.model = null;
        this.classNames = [];
        this.currentRound = 0;
        this.score = 0;
        this.timeLeft = 20;
        this.timer = null;
        this.predictionInterval = null;
        this.roundHistory = [];
        this.maxRounds = 5;
        this.currentWord = '';
        this.hasWon = false;
        
        // Model classes
        this.classNames = [
            'cat', 'dog', 'tree', 'car', 'cloud',
            'house', 'star', 'airplane', 'rainbow', 'chair'
        ];
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        await this.loadModel();
    }
    
    setupEventListeners() {
        // Start button
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startGame();
        });
        
        // Play again button
        document.getElementById('playAgainBtn').addEventListener('click', () => {
            this.startGame();
        });
        
        // Home button
        document.getElementById('homeBtn').addEventListener('click', () => {
            this.showScreen('startScreen');
        });
    }
    
    async loadModel() {
        try {
            // Show loading overlay
            document.getElementById('loadingOverlay').classList.remove('hidden');
            
            // Simple direct model loading
            const modelPath = './model/model.json';
            
            // First attempt: try standard loading
            try {
                this.model = await tf.loadLayersModel(modelPath);
                console.log('Model loaded successfully with standard loader');
            } catch (firstError) {
                console.log('Standard loading failed, trying with fixed config...');
                
                // Second attempt: fix the model configuration
                const response = await fetch(modelPath);
                const modelJson = await response.json();
                
                // Fix the InputLayer configuration
                if (modelJson.modelTopology && modelJson.modelTopology.model_config) {
                    const config = modelJson.modelTopology.model_config.config;
                    if (config.layers && config.layers[0]) {
                        const firstLayer = config.layers[0];
                        if (firstLayer.class_name === 'InputLayer' && firstLayer.config) {
                            // Fix batch_input_shape
                            if (!firstLayer.config.batch_input_shape && firstLayer.config.batch_shape) {
                                firstLayer.config.batch_input_shape = firstLayer.config.batch_shape;
                                delete firstLayer.config.batch_shape;
                            }
                        }
                    }
                    
                    // Fix build_input_shape if needed
                    if (!config.build_input_shape && config.layers[0].config.batch_input_shape) {
                        config.build_input_shape = config.layers[0].config.batch_input_shape;
                    }
                }
                
                // Fix weight names if needed
                if (modelJson.weightsManifest) {
                    modelJson.weightsManifest.forEach(group => {
                        if (group.weights) {
                            group.weights.forEach(weight => {
                                // Ensure weight names don't have 'sequential/' prefix if not needed
                                if (weight.name && !weight.name.startsWith('conv2d')) {
                                    // Keep the name as is
                                } else if (weight.name && !weight.name.includes('/')) {
                                    weight.name = 'sequential/' + weight.name;
                                }
                            });
                        }
                    });
                }
                
                // Save the modified model.json temporarily
                const modifiedModelJson = JSON.stringify(modelJson);
                const blob = new Blob([modifiedModelJson], { type: 'application/json' });
                const blobUrl = URL.createObjectURL(blob);
                
                // Create custom IOHandler
                const customIOHandler = {
                    load: async () => {
                        // Load weights
                        const weightResponse = await fetch('./model/group1-shard1of1.bin');
                        const weightData = await weightResponse.arrayBuffer();
                        
                        return {
                            modelTopology: modelJson.modelTopology,
                            weightSpecs: modelJson.weightsManifest[0].weights,
                            weightData: weightData,
                            format: modelJson.format,
                            generatedBy: modelJson.generatedBy,
                            convertedBy: modelJson.convertedBy
                        };
                    }
                };
                
                // Load model with custom handler
                this.model = await tf.loadLayersModel(customIOHandler);
                console.log('Model loaded with custom handler');
                
                // Clean up
                URL.revokeObjectURL(blobUrl);
            }
            
            console.log('Model input shape:', this.model.inputs[0].shape);
            console.log('Model output shape:', this.model.outputs[0].shape);
            
            // Load class names from file
            await this.loadClassNames();
            
        } catch (error) {
            console.error('Error loading model:', error);
            alert(`Error loading AI model: ${error.message}`);
            // Disable start button if model fails to load
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').textContent = 'Model Loading Failed';
        } finally {
            // Hide loading overlay
            document.getElementById('loadingOverlay').classList.add('hidden');
        }
    }
    
    async loadClassNames() {
        try {
            // Load class names from JSON file
            const response = await fetch('./model/classes.json');
            if (response.ok) {
                const data = await response.json();
                this.classNames = data.classes;
                console.log('Loaded classes:', this.classNames);
            } else {
                throw new Error('Classes file not found');
            }
        } catch (error) {
            console.error('Could not load class names:', error);
            // Use hardcoded classes as fallback
            this.classNames = [
                'cat', 'dog', 'tree', 'car', 'cloud',
                'house', 'star', 'airplane', 'rainbow', 'chair'
            ];
        }
    }
    
    startGame() {
        this.currentRound = 0;
        this.score = 0;
        this.roundHistory = [];
        this.updateScore();
        this.showScreen('gameScreen');
        this.nextRound();
    }
    
    nextRound() {
        if (this.currentRound >= this.maxRounds) {
            this.endGame();
            return;
        }
        
        this.currentRound++;
        this.hasWon = false;
        this.timeLeft = 20;
        
        // Clear the canvas
        if (drawingCanvas) {
            drawingCanvas.clear();
        }
        
        // Select a random word to draw
        this.currentWord = this.classNames[Math.floor(Math.random() * this.classNames.length)];
        document.getElementById('drawPrompt').textContent = this.currentWord;
        
        // Clear predictions
        document.getElementById('predictions').innerHTML = '<div class="prediction-item">Start drawing!</div>';
        
        // Start timer
        this.startTimer();
        
        // Start prediction loop
        this.startPredictions();
    }
    
    startTimer() {
        // Clear any existing timer
        if (this.timer) {
            clearInterval(this.timer);
        }
        
        this.updateTimerDisplay();
        
        this.timer = setInterval(() => {
            this.timeLeft--;
            this.updateTimerDisplay();
            
            if (this.timeLeft <= 0) {
                this.roundTimeout();
            }
        }, 1000);
    }
    
    updateTimerDisplay() {
        const timerElement = document.getElementById('timer');
        timerElement.textContent = this.timeLeft;
        
        // Change color based on time left
        if (this.timeLeft <= 5) {
            timerElement.style.color = '#e74c3c';
        } else if (this.timeLeft <= 10) {
            timerElement.style.color = '#f39c12';
        } else {
            timerElement.style.color = '#27ae60';
        }
    }
    
    startPredictions() {
        // Clear any existing prediction interval
        if (this.predictionInterval) {
            clearInterval(this.predictionInterval);
        }
        
        // Make predictions every 500ms
        this.predictionInterval = setInterval(() => {
            if (drawingCanvas && !drawingCanvas.isEmpty() && !this.hasWon) {
                this.makePrediction();
            }
        }, 500);
    }
    
    async makePrediction() {
        if (!drawingCanvas || this.hasWon || !this.model) return;
        
        try {
            // Get preprocessed image data (already normalized 0-1)
            const imageData = drawingCanvas.preprocessForModel(28); // 28x28 input
            
            const prediction = tf.tidy(() => {
                // Create tensor and reshape for the model [batch, height, width, channels]
                const img = tf.tensor(imageData, [28, 28, 1]).reshape([1, 28, 28, 1]);
                
                // Model expects float32 input normalized to 0-1 (already done in preprocessing)
                const floatImg = img.asType('float32');
                
                // Run prediction
                return this.model.predict(floatImg);
            });
            
            const scores = await prediction.array();
            prediction.dispose();
            
            // Get top 5 predictions from softmax output
            const topIndices = this.getTopIndices(scores[0], 5);
            const predictions = topIndices.map(idx => ({
                className: this.classNames[idx],
                probability: scores[0][idx]
            }));
            
            // Display predictions
            this.displayPredictions(predictions);
            
            // Check if the correct answer is in top predictions
            const correctPrediction = predictions.find(p => 
                p.className.toLowerCase() === this.currentWord.toLowerCase()
            );
            
            // Lower threshold for better gameplay experience
            if (correctPrediction && correctPrediction.probability > 0.3) {
                this.roundSuccess();
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
        }
    }
    
    
    getTopIndices(array, n) {
        const indices = array.map((val, idx) => ({val, idx}))
            .sort((a, b) => b.val - a.val)
            .slice(0, n)
            .map(item => item.idx);
        return indices;
    }
    
    displayPredictions(predictions) {
        const predictionsDiv = document.getElementById('predictions');
        predictionsDiv.innerHTML = '';
        
        predictions.forEach((pred, index) => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            
            // Highlight if it matches the current word
            if (pred.className.toLowerCase() === this.currentWord.toLowerCase()) {
                predItem.classList.add('correct');
            }
            
            const percentage = (pred.probability * 100).toFixed(1);
            predItem.textContent = `${index + 1}. ${pred.className} (${percentage}%)`;
            predictionsDiv.appendChild(predItem);
        });
    }
    
    roundSuccess() {
        if (this.hasWon) return;
        
        this.hasWon = true;
        this.score += Math.max(10, this.timeLeft * 2); // More points for faster completion
        this.updateScore();
        
        // Stop timers
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        // Show success feedback
        this.showFeedback('Great job! 🎉', 'success');
        
        // Record round
        this.roundHistory.push({
            word: this.currentWord,
            success: true,
            timeUsed: 20 - this.timeLeft
        });
        
        // Next round after delay
        setTimeout(() => {
            this.nextRound();
        }, 2000);
    }
    
    roundTimeout() {
        // Stop timers
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        // Show timeout feedback
        this.showFeedback('Time\'s up! ⏰', 'timeout');
        
        // Record round
        this.roundHistory.push({
            word: this.currentWord,
            success: false,
            timeUsed: 20
        });
        
        // Next round after delay
        setTimeout(() => {
            this.nextRound();
        }, 2000);
    }
    
    showFeedback(message, type) {
        const feedback = document.getElementById('feedback');
        feedback.textContent = message;
        feedback.className = `feedback show ${type}`;
        
        setTimeout(() => {
            feedback.classList.remove('show');
        }, 1500);
    }
    
    updateScore() {
        document.getElementById('score').textContent = this.score;
    }
    
    endGame() {
        // Stop any running timers
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        // Show results
        document.getElementById('finalScore').textContent = this.score;
        document.getElementById('roundsPlayed').textContent = this.currentRound;
        
        // Generate round summary
        const summaryDiv = document.getElementById('roundSummary');
        summaryDiv.innerHTML = '<h3>Round Summary:</h3>';
        
        this.roundHistory.forEach((round, index) => {
            const roundItem = document.createElement('div');
            roundItem.className = `round-item ${round.success ? 'success' : 'failed'}`;
            roundItem.innerHTML = `
                <span>Round ${index + 1}: ${round.word}</span>
                <span>${round.success ? '✓' : '✗'} (${round.timeUsed}s)</span>
            `;
            summaryDiv.appendChild(roundItem);
        });
        
        this.showScreen('resultsScreen');
    }
    
    showScreen(screenId) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show specified screen
        document.getElementById(screenId).classList.add('active');
    }
}

// Initialize game when DOM is loaded
let game;

document.addEventListener('DOMContentLoaded', () => {
    game = new QuickDrawGame();
});
