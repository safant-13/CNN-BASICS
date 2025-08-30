// Fixed game logic for Quick Draw Clone
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
            
            console.log('Loading model...');
            
            // Load model JSON
            const modelResponse = await fetch('./model/model.json');
            const modelJson = await modelResponse.json();
            
            // Load weights
            const weightsResponse = await fetch('./model/group1-shard1of1.bin');
            const weightData = await weightsResponse.arrayBuffer();
            
            // Fix the model configuration
            const fixedModelJson = JSON.parse(JSON.stringify(modelJson));
            
            // Fix InputLayer configuration
            if (fixedModelJson.modelTopology && 
                fixedModelJson.modelTopology.model_config && 
                fixedModelJson.modelTopology.model_config.config) {
                
                const config = fixedModelJson.modelTopology.model_config.config;
                const layers = config.layers;
                
                if (layers && layers[0] && layers[0].class_name === 'InputLayer') {
                    const inputLayer = layers[0];
                    if (inputLayer.config) {
                        // Set batch_input_shape properly
                        inputLayer.config.batch_input_shape = [null, 28, 28, 1];
                        delete inputLayer.config.batch_shape;
                    }
                }
                
                // IMPORTANT: Change the model name to match weight prefixes
                // The weights are prefixed with "sequential/" so the model name should be "sequential"
                config.name = 'sequential';
            }
            
            // Create custom IO handler
            const customIOHandler = {
                load: async () => {
                    console.log('Custom loader: Loading model with fixed configuration');
                    
                    return {
                        modelTopology: fixedModelJson.modelTopology,
                        weightSpecs: fixedModelJson.weightsManifest[0].weights,
                        weightData: weightData,
                        format: fixedModelJson.format,
                        generatedBy: fixedModelJson.generatedBy,
                        convertedBy: fixedModelJson.convertedBy,
                        userDefinedMetadata: fixedModelJson.userDefinedMetadata
                    };
                }
            };
            
            // Load the model
            this.model = await tf.loadLayersModel(customIOHandler);
            
            console.log('Model loaded successfully!');
            console.log('Model name:', this.model.name);
            console.log('Input shape:', this.model.inputs[0].shape);
            console.log('Output shape:', this.model.outputs[0].shape);
            
            // Test prediction
            const testInput = tf.zeros([1, 28, 28, 1]);
            const testOutput = this.model.predict(testInput);
            const testResult = await testOutput.array();
            console.log('Test prediction successful!');
            console.log('Output values (should sum to ~1):', testResult[0]);
            console.log('Sum of outputs:', testResult[0].reduce((a, b) => a + b, 0));
            testInput.dispose();
            testOutput.dispose();
            
        } catch (error) {
            console.error('Error loading model:', error);
            
            // Try alternative: rename weights to remove sequential/ prefix
            try {
                console.log('Trying alternative approach...');
                
                const modelResponse = await fetch('./model/model.json');
                const modelJson = await modelResponse.json();
                
                const weightsResponse = await fetch('./model/group1-shard1of1.bin');
                const weightData = await weightsResponse.arrayBuffer();
                
                // Create a modified version
                const altModelJson = JSON.parse(JSON.stringify(modelJson));
                
                // Fix InputLayer
                if (altModelJson.modelTopology?.model_config?.config?.layers?.[0]) {
                    const inputLayer = altModelJson.modelTopology.model_config.config.layers[0];
                    if (inputLayer.class_name === 'InputLayer' && inputLayer.config) {
                        inputLayer.config.batch_input_shape = [null, 28, 28, 1];
                        delete inputLayer.config.batch_shape;
                    }
                }
                
                // Remove 'sequential/' prefix from weight names
                if (altModelJson.weightsManifest?.[0]?.weights) {
                    altModelJson.weightsManifest[0].weights = altModelJson.weightsManifest[0].weights.map(weight => ({
                        ...weight,
                        name: weight.name.replace('sequential/', '')
                    }));
                    
                    console.log('Renamed weights:', altModelJson.weightsManifest[0].weights.map(w => w.name));
                }
                
                // Also update the model name to not be 'sequential'
                if (altModelJson.modelTopology?.model_config?.config) {
                    altModelJson.modelTopology.model_config.config.name = 'model';
                }
                
                const altIOHandler = {
                    load: async () => ({
                        modelTopology: altModelJson.modelTopology,
                        weightSpecs: altModelJson.weightsManifest[0].weights,
                        weightData: weightData,
                        format: altModelJson.format,
                        generatedBy: altModelJson.generatedBy,
                        convertedBy: altModelJson.convertedBy
                    })
                };
                
                this.model = await tf.loadLayersModel(altIOHandler);
                console.log('Model loaded with alternative approach!');
                
                // Test the model after alternative loading
                console.log('Model name:', this.model.name);
                console.log('Input shape:', this.model.inputs[0].shape);
                console.log('Output shape:', this.model.outputs[0].shape);
                
                const testInput = tf.zeros([1, 28, 28, 1]);
                const testOutput = this.model.predict(testInput);
                const testResult = await testOutput.array();
                console.log('Test prediction successful!');
                console.log('Output values:', testResult[0]);
                console.log('Sum of outputs:', testResult[0].reduce((a, b) => a + b, 0));
                testInput.dispose();
                testOutput.dispose();
                
            } catch (altError) {
                console.error('Alternative approach also failed:', altError);
                alert('Failed to load the model. Please check the console for details.');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('startBtn').textContent = 'Model Loading Failed';
            }
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
            // Use inverted colors (true) since Quick Draw uses white on black
            const imageData = drawingCanvas.preprocessForModel(28, true);
            
            // Debug: Log first few values of preprocessed data
            if (Math.random() < 0.1) { // Log 10% of the time to avoid spam
                console.log('Sample preprocessed values (first 10):', 
                    Array.from(imageData.slice(0, 10)).map(v => v.toFixed(3)));
                console.log('Non-zero pixels (white strokes):', 
                    Array.from(imageData).filter(v => v > 0.1).length);
                
                // Show a mini preview of what the model sees
                const preview = [];
                for (let y = 0; y < 28; y += 7) {
                    let row = '';
                    for (let x = 0; x < 28; x += 7) {
                        const val = imageData[y * 28 + x];
                        row += val > 0.5 ? '█' : val > 0.2 ? '▒' : '░';
                    }
                    preview.push(row);
                }
                console.log('Mini preview (28x28 downsampled):\n' + preview.join('\n'));
            }
            
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
            
            // Debug: Log top prediction
            if (predictions[0].probability > 0.5) {
                console.log(`High confidence: ${predictions[0].className} (${(predictions[0].probability * 100).toFixed(1)}%)`);
            }
            
            const correctPrediction = predictions.find(p => 
                p.className.toLowerCase() === this.currentWord.toLowerCase()
            );
            
            // Lower threshold for testing
            if (correctPrediction && correctPrediction.probability > 0.2) {
                console.log(`Correct prediction found: ${correctPrediction.className} with ${(correctPrediction.probability * 100).toFixed(1)}%`);
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
