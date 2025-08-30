// Simplified game logic for Quick Draw Clone
class QuickDrawGame {
    constructor() {
        this.model = null;
        this.classNames = [
            'cat', 'dog', 'tree', 'car', 'cloud',
            'house', 'star', 'airplane', 'rainbow', 'chair'
        ];
        this.currentRound = 0;
        this.score = 0;
        this.timeLeft = 20;
        this.timer = null;
        this.predictionInterval = null;
        this.roundHistory = [];
        this.maxRounds = 5;
        this.currentWord = '';
        this.hasWon = false;
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        await this.loadModel();
    }
    
    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startGame();
        });
        
        document.getElementById('playAgainBtn').addEventListener('click', () => {
            this.startGame();
        });
        
        document.getElementById('homeBtn').addEventListener('click', () => {
            this.showScreen('startScreen');
        });
    }
    
    async loadModel() {
        try {
            document.getElementById('loadingOverlay').classList.remove('hidden');
            
            console.log('Starting model load...');
            
            // First, try the simplest approach - direct loading
            try {
                this.model = await tf.loadLayersModel('./model/model.json');
                console.log('Model loaded directly!');
            } catch (directError) {
                console.log('Direct loading failed:', directError.message);
                console.log('Trying custom loader...');
                
                // Load model JSON and weights manually
                const modelResponse = await fetch('./model/model.json');
                const modelJson = await modelResponse.json();
                
                // Load weights binary
                const weightsResponse = await fetch('./model/group1-shard1of1.bin');
                const weightData = await weightsResponse.arrayBuffer();
                
                // Create a custom IO handler
                const customIOHandler = {
                    load: async () => {
                        // Fix the model topology if needed
                        const topology = modelJson.modelTopology;
                        
                        // Fix InputLayer if needed
                        if (topology && topology.model_config && topology.model_config.config) {
                            const layers = topology.model_config.config.layers;
                            if (layers && layers[0] && layers[0].class_name === 'InputLayer') {
                                const inputLayer = layers[0];
                                if (inputLayer.config) {
                                    // Ensure batch_input_shape is set
                                    if (!inputLayer.config.batch_input_shape && inputLayer.config.batch_shape) {
                                        inputLayer.config.batch_input_shape = inputLayer.config.batch_shape;
                                    } else if (!inputLayer.config.batch_input_shape) {
                                        inputLayer.config.batch_input_shape = [null, 28, 28, 1];
                                    }
                                    // Remove batch_shape to avoid confusion
                                    delete inputLayer.config.batch_shape;
                                }
                            }
                        }
                        
                        // Return the model configuration
                        return {
                            modelTopology: topology,
                            weightSpecs: modelJson.weightsManifest[0].weights,
                            weightData: weightData,
                            format: modelJson.format,
                            generatedBy: modelJson.generatedBy,
                            convertedBy: modelJson.convertedBy
                        };
                    }
                };
                
                this.model = await tf.loadLayersModel(customIOHandler);
                console.log('Model loaded with custom IO handler!');
            }
            
            console.log('Model loaded successfully!');
            console.log('Input shape:', this.model.inputs[0].shape);
            console.log('Output shape:', this.model.outputs[0].shape);
            
            // Test the model with dummy input
            const testInput = tf.zeros([1, 28, 28, 1]);
            const testOutput = this.model.predict(testInput);
            const testResult = await testOutput.array();
            console.log('Test prediction successful, output length:', testResult[0].length);
            console.log('Classes available:', this.classNames);
            testInput.dispose();
            testOutput.dispose();
            
        } catch (error) {
            console.error('Error loading model:', error);
            alert(`Error loading AI model: ${error.message}\n\nPlease check the browser console for details.`);
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').textContent = 'Model Loading Failed';
        } finally {
            document.getElementById('loadingOverlay').classList.add('hidden');
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
        
        if (drawingCanvas) {
            drawingCanvas.clear();
        }
        
        this.currentWord = this.classNames[Math.floor(Math.random() * this.classNames.length)];
        document.getElementById('drawPrompt').textContent = this.currentWord;
        document.getElementById('predictions').innerHTML = '<div class="prediction-item">Start drawing!</div>';
        
        this.startTimer();
        this.startPredictions();
    }
    
    startTimer() {
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
        
        if (this.timeLeft <= 5) {
            timerElement.style.color = '#e74c3c';
        } else if (this.timeLeft <= 10) {
            timerElement.style.color = '#f39c12';
        } else {
            timerElement.style.color = '#27ae60';
        }
    }
    
    startPredictions() {
        if (this.predictionInterval) {
            clearInterval(this.predictionInterval);
        }
        
        this.predictionInterval = setInterval(() => {
            if (drawingCanvas && !drawingCanvas.isEmpty() && !this.hasWon) {
                this.makePrediction();
            }
        }, 500);
    }
    
    async makePrediction() {
        if (!drawingCanvas || this.hasWon || !this.model) return;
        
        try {
            const imageData = drawingCanvas.preprocessForModel(28);
            
            const prediction = tf.tidy(() => {
                const img = tf.tensor(imageData, [28, 28, 1]).reshape([1, 28, 28, 1]);
                return this.model.predict(img);
            });
            
            const scores = await prediction.array();
            prediction.dispose();
            
            const topIndices = this.getTopIndices(scores[0], 5);
            const predictions = topIndices.map(idx => ({
                className: this.classNames[idx],
                probability: scores[0][idx]
            }));
            
            this.displayPredictions(predictions);
            
            const correctPrediction = predictions.find(p => 
                p.className.toLowerCase() === this.currentWord.toLowerCase()
            );
            
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
        this.score += Math.max(10, this.timeLeft * 2);
        this.updateScore();
        
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        this.showFeedback('Great job! 🎉', 'success');
        
        this.roundHistory.push({
            word: this.currentWord,
            success: true,
            timeUsed: 20 - this.timeLeft
        });
        
        setTimeout(() => {
            this.nextRound();
        }, 2000);
    }
    
    roundTimeout() {
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        this.showFeedback('Time\'s up! ⏰', 'timeout');
        
        this.roundHistory.push({
            word: this.currentWord,
            success: false,
            timeUsed: 20
        });
        
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
        clearInterval(this.timer);
        clearInterval(this.predictionInterval);
        
        document.getElementById('finalScore').textContent = this.score;
        document.getElementById('roundsPlayed').textContent = this.currentRound;
        
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
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        document.getElementById(screenId).classList.add('active');
    }
}

// Initialize game when DOM is loaded
let game;

document.addEventListener('DOMContentLoaded', () => {
    game = new QuickDrawGame();
});
