// Fixed preprocessing function for QuickDraw game
// This should replace the preprocessCanvas function in your game.js

function preprocessCanvas(canvas) {
    // Get the canvas context
    const ctx = canvas.getContext('2d');
    
    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with white background first (matching QuickDraw format)
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    // Draw the resized canvas (black strokes on white background)
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convert to grayscale and normalize
    const grayscale = [];
    for (let i = 0; i < data.length; i += 4) {
        // Convert RGB to grayscale using standard formula
        // QuickDraw uses black strokes (0) on white background (255)
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        
        // Normalize to [0, 1] range
        // White (255) becomes 1.0, Black (0) becomes 0.0
        const normalized = gray / 255.0;
        grayscale.push(normalized);
    }
    
    // Debug logging (can be removed in production)
    if (window.debugMode) {
        console.log('Preprocessing stats:');
        const min = Math.min(...grayscale);
        const max = Math.max(...grayscale);
        const mean = grayscale.reduce((a, b) => a + b, 0) / grayscale.length;
        console.log(`  Min: ${min.toFixed(3)}, Max: ${max.toFixed(3)}, Mean: ${mean.toFixed(3)}`);
        
        // Visual debug - show first few rows
        console.log('First 5 rows (simplified):');
        for (let row = 0; row < 5; row++) {
            let rowStr = '';
            for (let col = 0; col < 28; col++) {
                const val = grayscale[row * 28 + col];
                if (val < 0.2) rowStr += '█'; // Black (drawn)
                else if (val < 0.5) rowStr += '▓';
                else if (val < 0.8) rowStr += '░';
                else rowStr += ' '; // White (background)
            }
            console.log(rowStr);
        }
    }
    
    // Reshape to match model input: [1, 28, 28, 1]
    // TensorFlow.js expects a 4D tensor
    const tensor = tf.tensor4d(grayscale, [1, 28, 28, 1]);
    
    return tensor;
}

// Alternative version if your canvas drawing is inverted (white on black)
function preprocessCanvasWithInversion(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Create temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with white background
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    // Draw resized
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convert and check if inversion needed
    const grayscale = [];
    let sum = 0;
    
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        grayscale.push(gray);
        sum += gray;
    }
    
    const mean = sum / grayscale.length;
    
    // If mean is low (mostly black), we have white strokes on black background
    // Need to invert to match QuickDraw format (black strokes on white)
    const needsInversion = mean < 128;
    
    const normalized = grayscale.map(val => {
        if (needsInversion) {
            val = 255 - val; // Invert
        }
        return val / 255.0; // Normalize to [0, 1]
    });
    
    // Create tensor
    const tensor = tf.tensor4d(normalized, [1, 28, 28, 1]);
    
    return tensor;
}

// Fixed prediction function
async function predict(canvas) {
    if (!model) {
        console.error('Model not loaded');
        return null;
    }
    
    try {
        // Preprocess the canvas
        const input = preprocessCanvas(canvas);
        
        // Make prediction
        const prediction = await model.predict(input).data();
        
        // Clean up tensor
        input.dispose();
        
        // Get top predictions
        const topIndex = prediction.indexOf(Math.max(...prediction));
        const confidence = prediction[topIndex];
        
        // Return result
        return {
            class: classes[topIndex],
            confidence: confidence,
            allPredictions: classes.map((cls, idx) => ({
                class: cls,
                confidence: prediction[idx]
            })).sort((a, b) => b.confidence - a.confidence)
        };
        
    } catch (error) {
        console.error('Prediction error:', error);
        return null;
    }
}

// Canvas setup fix - ensure white background
function initCanvas() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing style for black strokes
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 8; // Thicker line for better recognition
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    return { canvas, ctx };
}

// Export for use in your game
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        preprocessCanvas,
        preprocessCanvasWithInversion,
        predict,
        initCanvas
    };
}
