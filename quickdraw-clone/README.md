# Quick Draw Clone 🎨

A web-based drawing game inspired by Google's Quick, Draw! where an AI tries to guess what you're drawing in real-time.

## Features

- **Real-time Drawing**: Smooth canvas drawing with mouse and touch support
- **AI Predictions**: TensorFlow.js integration for real-time drawing recognition
- **Game Mechanics**: 20-second timer, score tracking, and multiple rounds
- **Responsive Design**: Works on desktop and mobile devices
- **10 Drawing Categories**: cat, dog, tree, car, cloud, house, star, airplane, rainbow, chair

## Project Structure

```
quickdraw-clone/
│
├── index.html          # Main HTML file
├── styles.css          # Styling and animations
├── drawing.js          # Canvas drawing functionality
├── game.js            # Game logic and TensorFlow.js integration
├── model/             # Directory for TensorFlow.js model files
│   ├── README.md      # Instructions for adding models
│   └── classes.json.example  # Example class names file
└── README.md          # This file
```

## Getting Started

### 1. Run the Game

The easiest way to run the game is using a local web server:

**Using Python:**
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

**Using Node.js:**
```bash
# Install http-server globally
npm install -g http-server

# Run server
http-server
```

Then open your browser and navigate to `http://localhost:8000`

### 2. Model Configuration

The game is configured to work with the following 10 drawing categories:
- cat
- dog
- tree
- car
- cloud
- house
- star
- airplane
- rainbow
- chair

The TensorFlow.js model files are already in place in the `model/` directory.

### 3. Customize the Game

You can modify various aspects of the game:

- **Input Size**: Change the model input size in `drawing.js` (line ~148):
  ```javascript
  const imageData = drawingCanvas.preprocessForModel(28); // Change 28 to your model's input size
  ```

- **Game Duration**: Modify the timer in `game.js` (line ~9):
  ```javascript
  this.timeLeft = 20; // Change to desired seconds
  ```

- **Number of Rounds**: Change max rounds in `game.js` (line ~12):
  ```javascript
  this.maxRounds = 5; // Change to desired number of rounds
  ```

## How It Works

1. **Drawing Canvas**: Users draw on an HTML5 canvas element
2. **Preprocessing**: Drawings are converted to grayscale and resized to match model input
3. **Prediction**: TensorFlow.js model processes the image and returns class probabilities
4. **Game Logic**: Checks if the AI correctly identifies the drawing within time limit
5. **Scoring**: Points awarded based on speed and accuracy

## Game Modes

The game uses your trained TensorFlow.js model to recognize drawings in real-time. The model processes 28x28 grayscale images and predicts which of the 10 categories you're drawing.

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers with touch support

## Technologies Used

- HTML5 Canvas API
- TensorFlow.js
- Vanilla JavaScript
- CSS3 animations

## Tips for Best Results

1. Draw clearly and use most of the canvas space
2. Complete your drawing - partial drawings are harder to recognize
3. The AI makes predictions every 500ms, so keep drawing!
4. Simple, iconic representations work better than detailed drawings

## Troubleshooting

- **Model not loading**: Check browser console for errors, ensure model files are in correct directory
- **Predictions not working**: Verify model input/output dimensions match the preprocessing
- **Canvas not responsive**: Check if JavaScript is enabled in your browser

## License

This is a demonstration project for educational purposes.

---

Have fun drawing! 🖌️✨
