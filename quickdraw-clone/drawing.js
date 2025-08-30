// Drawing functionality for Quick Draw Clone
class DrawingCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.strokeHistory = [];
        this.currentStroke = [];
        this.lastX = 0;
        this.lastY = 0;
        
        this.setupCanvas();
        this.setupEventListeners();
    }
    
    setupCanvas() {
        // Set canvas size
        this.canvas.width = 400;
        this.canvas.height = 400;
        
        // Configure drawing style - Match Quick Draw dataset
        // Quick Draw uses thicker lines (5-8 pixels typically)
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 5;  // Increased from 3 to match Quick Draw
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        // Fill with white background
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Enable smoothing
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
        
        // Prevent scrolling when touching the canvas
        this.canvas.addEventListener('touchstart', (e) => e.preventDefault());
        this.canvas.addEventListener('touchmove', (e) => e.preventDefault());
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.currentStroke = [];
        
        const rect = this.canvas.getBoundingClientRect();
        this.lastX = e.clientX - rect.left;
        this.lastY = e.clientY - rect.top;
        
        this.currentStroke.push({x: this.lastX, y: this.lastY});
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        // Smooth line drawing with quadratic curves for better quality
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        
        // Use quadratic curve for smoother lines
        const midX = (this.lastX + currentX) / 2;
        const midY = (this.lastY + currentY) / 2;
        this.ctx.quadraticCurveTo(this.lastX, this.lastY, midX, midY);
        
        this.ctx.stroke();
        
        this.currentStroke.push({x: currentX, y: currentY});
        
        this.lastX = currentX;
        this.lastY = currentY;
    }
    
    stopDrawing() {
        if (this.isDrawing && this.currentStroke.length > 0) {
            this.strokeHistory.push([...this.currentStroke]);
        }
        this.isDrawing = false;
        this.currentStroke = [];
    }
    
    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                         e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    clear() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.strokeHistory = [];
        this.currentStroke = [];
    }
    
    undo() {
        if (this.strokeHistory.length === 0) return;
        
        this.strokeHistory.pop();
        this.redrawCanvas();
    }
    
    redrawCanvas() {
        // Clear canvas
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Redraw all strokes
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 5;  // Match the thicker line width
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        for (const stroke of this.strokeHistory) {
            if (stroke.length < 2) continue;
            
            this.ctx.beginPath();
            this.ctx.moveTo(stroke[0].x, stroke[0].y);
            
            // Use smooth curves for redrawing
            for (let i = 1; i < stroke.length - 1; i++) {
                const midX = (stroke[i].x + stroke[i + 1].x) / 2;
                const midY = (stroke[i].y + stroke[i + 1].y) / 2;
                this.ctx.quadraticCurveTo(stroke[i].x, stroke[i].y, midX, midY);
            }
            
            // Draw the last segment
            if (stroke.length > 1) {
                const lastPoint = stroke[stroke.length - 1];
                this.ctx.lineTo(lastPoint.x, lastPoint.y);
            }
            
            this.ctx.stroke();
        }
    }
    
    getImageData() {
        // Get the image data for model prediction
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return imageData;
    }
    
    getCanvasAsImage() {
        // Convert canvas to base64 image
        return this.canvas.toDataURL('image/png');
    }
    
    preprocessForModel(targetSize = 28, invertColors = true) {
        // Create a temporary canvas for resizing
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = targetSize;
        tempCanvas.height = targetSize;
        
        // Enable smoothing for better downscaling
        tempCtx.imageSmoothingEnabled = true;
        tempCtx.imageSmoothingQuality = 'high';
        
        // Quick Draw dataset typically has black background with white strokes
        // But we draw black on white, so we'll handle the inversion
        
        if (invertColors) {
            // Fill with BLACK background (like Quick Draw dataset)
            tempCtx.fillStyle = 'black';
            tempCtx.fillRect(0, 0, targetSize, targetSize);
            
            // Draw the image inverted
            tempCtx.globalCompositeOperation = 'difference';
            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, targetSize, targetSize);
            tempCtx.globalCompositeOperation = 'source-over';
        } else {
            // Fill with white background
            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, targetSize, targetSize);
        }
        
        // Draw the current canvas scaled down to exactly 28x28
        tempCtx.drawImage(this.canvas, 0, 0, this.canvas.width, this.canvas.height, 0, 0, targetSize, targetSize);
        
        // If inverting, apply the inversion
        if (invertColors) {
            tempCtx.globalCompositeOperation = 'difference';
            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, targetSize, targetSize);
            tempCtx.globalCompositeOperation = 'source-over';
        }
        
        // Get the scaled image data
        const imageData = tempCtx.getImageData(0, 0, targetSize, targetSize);
        const data = imageData.data;
        
        // Convert to grayscale float32 array normalized to 0-1
        const grayscale = new Float32Array(targetSize * targetSize);
        
        for (let i = 0; i < data.length; i += 4) {
            // Get RGB values
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Convert to grayscale
            let grayValue = 0.299 * r + 0.587 * g + 0.114 * b;
            
            // Quick Draw dataset has white strokes on black background
            // So white (255) should be 1.0 and black (0) should be 0.0
            // This matches what the model expects
            
            // Normalize to 0-1 range
            grayscale[i / 4] = grayValue / 255.0;
        }
        
        return grayscale;
    }
    
    isEmpty() {
        // Check if canvas has any drawings
        return this.strokeHistory.length === 0;
    }
}

// Initialize drawing canvas when DOM is loaded
let drawingCanvas;

document.addEventListener('DOMContentLoaded', () => {
    drawingCanvas = new DrawingCanvas('drawingCanvas');
    
    // Setup control buttons
    const clearBtn = document.getElementById('clearBtn');
    const undoBtn = document.getElementById('undoBtn');
    
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            drawingCanvas.clear();
        });
    }
    
    if (undoBtn) {
        undoBtn.addEventListener('click', () => {
            drawingCanvas.undo();
        });
    }
});
