import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import time
import random

class QuickDrawGame:
    def __init__(self, model_path='quickdraw_cnn_model.h5', labels_path='labels.txt'):
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        self.CANVAS_SIZE = 400
        self.MODEL_INPUT_SIZE = 28
        
        # Colors (RGB)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (100, 149, 237)
        
        # Setup display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Quick, Draw! Clone")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Canvas setup - white background for drawing
        self.canvas_x = 50
        self.canvas_y = 100
        self.canvas = pygame.Surface((self.CANVAS_SIZE, self.CANVAS_SIZE))
        self.canvas.fill(self.WHITE)  # White background
        self.drawing = False
        self.last_pos = None
        self.brush_size = 8  # Thicker brush for better visibility
        
        # Load model and labels
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            
            # Load labels
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.labels)} labels: {self.labels}")
            
        except Exception as e:
            print(f"Error loading model or labels: {e}")
            self.model = None
            self.labels = []
        
        # Game state
        self.current_word = None
        self.time_limit = 20  # seconds per word
        self.start_time = None
        self.score = 0
        self.game_state = "menu"  # menu, playing, gameover
        self.prediction = None
        self.confidence = 0
        self.debug_mode = False  # Toggle with 'D' key
        
    def get_canvas_array(self):
        """Convert pygame surface to numpy array matching training data format"""
        # Get the canvas as a string buffer
        canvas_string = pygame.image.tostring(self.canvas, 'RGB')
        
        # Convert to numpy array
        canvas_array = np.frombuffer(canvas_string, dtype=np.uint8)
        canvas_array = canvas_array.reshape((self.CANVAS_SIZE, self.CANVAS_SIZE, 3))
        
        # Convert to grayscale using the same method as OpenCV
        # Use standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(canvas_array[...,:3], [0.299, 0.587, 0.114])
        gray = gray.astype(np.uint8)
        
        # Resize to model input size
        resized = cv2.resize(gray, (self.MODEL_INPUT_SIZE, self.MODEL_INPUT_SIZE), 
                            interpolation=cv2.INTER_AREA)
        
        return resized
    
    def preprocess_for_model(self, image_array):
        """Preprocess the drawing for model prediction to match training data format"""
        # The QuickDraw dataset format:
        # - Black strokes (drawing) = 0 (black)
        # - White background = 255 (white)
        # After normalization: black = 0.0, white = 1.0
        
        # Our canvas starts white (255) with black drawings (0)
        # This matches the QuickDraw format already!
        
        # Create a copy to avoid modifying the original
        processed = image_array.copy()
        
        # Ensure the image has good contrast
        # If the image is too uniform (user hasn't drawn much), skip processing
        if np.std(processed) < 10:  # Very low variance means little/no drawing
            # Return normalized empty canvas
            processed = processed.astype('float32') / 255.0
            processed = processed.reshape(1, 28, 28, 1)
            return processed
        
        # Apply slight Gaussian blur to smooth the lines (matching dataset characteristics)
        processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
        
        # Normalize to [0, 1] range
        processed = processed.astype('float32') / 255.0
        
        # Reshape for model input (batch_size=1, height=28, width=28, channels=1)
        processed = processed.reshape(1, 28, 28, 1)
        
        if self.debug_mode:
            # Debug: Print statistics
            print(f"Preprocessing stats:")
            print(f"  Shape: {processed.shape}")
            print(f"  Min value: {processed.min():.3f}")
            print(f"  Max value: {processed.max():.3f}")
            print(f"  Mean value: {processed.mean():.3f}")
            print(f"  Std dev: {processed.std():.3f}")
            
            # Show a text representation of the processed image
            debug_img = processed[0, :, :, 0]
            print("\nProcessed image preview (first 10x10 pixels):")
            for i in range(min(10, debug_img.shape[0])):
                row_str = ""
                for j in range(min(10, debug_img.shape[1])):
                    val = debug_img[i, j]
                    if val < 0.2:
                        row_str += "█"  # Black (drawn area)
                    elif val < 0.5:
                        row_str += "▓"
                    elif val < 0.8:
                        row_str += "░"
                    else:
                        row_str += " "  # White (background)
                row_str += f"  [{debug_img[i, :10].min():.2f}-{debug_img[i, :10].max():.2f}]"
                print(row_str)
        
        return processed
    
    def predict_drawing(self):
        """Make prediction on current drawing"""
        if self.model is None:
            return None, 0
        
        # Get canvas as numpy array
        canvas_array = self.get_canvas_array()
        
        # Preprocess for model
        processed = self.preprocess_for_model(canvas_array)
        
        # Make prediction
        try:
            predictions = self.model.predict(processed, verbose=0)
            
            # Get top prediction
            top_idx = np.argmax(predictions[0])
            confidence = predictions[0][top_idx]
            
            if top_idx < len(self.labels):
                label = self.labels[top_idx]
            else:
                label = f"Class {top_idx}"
            
            if self.debug_mode:
                # Show top 3 predictions
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                print("\nTop 3 predictions:")
                for idx in top_3_idx:
                    if idx < len(self.labels):
                        print(f"  {self.labels[idx]}: {predictions[0][idx]:.2%}")
            
            return label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0
    
    def start_new_round(self):
        """Start a new drawing round"""
        self.canvas.fill(self.WHITE)  # Clear canvas with white
        self.current_word = random.choice(self.labels) if self.labels else "cat"
        self.start_time = time.time()
        self.game_state = "playing"
        self.prediction = None
        self.confidence = 0
        print(f"\nNew round! Draw: {self.current_word}")
    
    def check_win_condition(self):
        """Check if player has won the round"""
        if self.prediction and self.prediction.lower() == self.current_word.lower():
            if self.confidence > 0.5:  # Require 50% confidence
                return True
        return False
    
    def handle_mouse_events(self, event):
        """Handle mouse input for drawing"""
        if self.game_state != "playing":
            return
            
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Check if mouse is within canvas bounds
        if (self.canvas_x <= mouse_x <= self.canvas_x + self.CANVAS_SIZE and
            self.canvas_y <= mouse_y <= self.canvas_y + self.CANVAS_SIZE):
            
            canvas_x = mouse_x - self.canvas_x
            canvas_y = mouse_y - self.canvas_y
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.drawing = True
                self.last_pos = (canvas_x, canvas_y)
                
            elif event.type == pygame.MOUSEBUTTONUP:
                self.drawing = False
                self.last_pos = None
                
            elif event.type == pygame.MOUSEMOTION and self.drawing:
                if self.last_pos:
                    # Draw with black color on white canvas
                    pygame.draw.line(self.canvas, self.BLACK, 
                                   self.last_pos, (canvas_x, canvas_y), 
                                   self.brush_size)
                    # Also draw a circle for smoother lines
                    pygame.draw.circle(self.canvas, self.BLACK, 
                                     (canvas_x, canvas_y), 
                                     self.brush_size // 2)
                self.last_pos = (canvas_x, canvas_y)
                
                # Make prediction while drawing
                self.prediction, self.confidence = self.predict_drawing()
    
    def draw_menu(self):
        """Draw the main menu"""
        self.screen.fill(self.BLUE)
        
        title = self.font.render("Quick, Draw! Clone", True, self.WHITE)
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH//2, 150))
        self.screen.blit(title, title_rect)
        
        if self.model:
            start_text = self.font.render("Press SPACE to Start", True, self.WHITE)
            start_rect = start_text.get_rect(center=(self.WINDOW_WIDTH//2, 300))
            self.screen.blit(start_text, start_rect)
            
            info_text = self.small_font.render(f"Model loaded with {len(self.labels)} categories", True, self.WHITE)
            info_rect = info_text.get_rect(center=(self.WINDOW_WIDTH//2, 350))
            self.screen.blit(info_text, info_rect)
        else:
            error_text = self.font.render("Model not loaded!", True, self.RED)
            error_rect = error_text.get_rect(center=(self.WINDOW_WIDTH//2, 300))
            self.screen.blit(error_text, error_rect)
        
        quit_text = self.small_font.render("Press Q to Quit", True, self.WHITE)
        quit_rect = quit_text.get_rect(center=(self.WINDOW_WIDTH//2, 450))
        self.screen.blit(quit_text, quit_rect)
    
    def draw_game(self):
        """Draw the game screen"""
        self.screen.fill(self.GRAY)
        
        # Draw canvas with border
        pygame.draw.rect(self.screen, self.BLACK, 
                        (self.canvas_x-2, self.canvas_y-2, 
                         self.CANVAS_SIZE+4, self.CANVAS_SIZE+4))
        self.screen.blit(self.canvas, (self.canvas_x, self.canvas_y))
        
        # Draw current word
        word_text = self.font.render(f"Draw: {self.current_word}", True, self.BLACK)
        self.screen.blit(word_text, (self.canvas_x, 40))
        
        # Draw timer
        if self.start_time:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.time_limit - elapsed)
            timer_color = self.RED if remaining < 5 else self.BLACK
            timer_text = self.font.render(f"Time: {remaining:.1f}s", True, timer_color)
            self.screen.blit(timer_text, (self.canvas_x + 300, 40))
            
            # Check if time's up
            if remaining <= 0:
                self.game_state = "gameover"
        
        # Draw prediction
        if self.prediction:
            pred_color = self.GREEN if self.check_win_condition() else self.BLACK
            pred_text = self.small_font.render(
                f"I think it's: {self.prediction} ({self.confidence:.1%})", 
                True, pred_color)
            self.screen.blit(pred_text, (self.canvas_x, self.canvas_y + self.CANVAS_SIZE + 20))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (self.WINDOW_WIDTH - 200, 40))
        
        # Draw instructions
        instructions = [
            "Draw with mouse",
            "C - Clear canvas",
            "SPACE - New word",
            "D - Toggle debug",
            "ESC - Menu"
        ]
        y_offset = 150
        for instruction in instructions:
            inst_text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(inst_text, (self.CANVAS_SIZE + 100, y_offset))
            y_offset += 30
        
        # Draw debug mode indicator
        if self.debug_mode:
            debug_text = self.small_font.render("DEBUG MODE ON", True, self.RED)
            self.screen.blit(debug_text, (self.CANVAS_SIZE + 100, y_offset + 30))
    
    def draw_gameover(self):
        """Draw game over screen"""
        self.screen.fill(self.BLUE)
        
        if self.check_win_condition():
            result_text = self.font.render("You Won!", True, self.GREEN)
            self.score += 1
        else:
            result_text = self.font.render("Time's Up!", True, self.RED)
        
        result_rect = result_text.get_rect(center=(self.WINDOW_WIDTH//2, 200))
        self.screen.blit(result_text, result_rect)
        
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.WINDOW_WIDTH//2, 300))
        self.screen.blit(score_text, score_rect)
        
        continue_text = self.small_font.render("Press SPACE for next round or ESC for menu", True, self.WHITE)
        continue_rect = continue_text.get_rect(center=(self.WINDOW_WIDTH//2, 400))
        self.screen.blit(continue_text, continue_rect)
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_ESCAPE:
                        self.game_state = "menu"
                    
                    elif event.key == pygame.K_SPACE:
                        if self.game_state == "menu" and self.model:
                            self.start_new_round()
                        elif self.game_state == "gameover":
                            self.start_new_round()
                        elif self.game_state == "playing":
                            self.start_new_round()  # Skip current word
                    
                    elif event.key == pygame.K_c and self.game_state == "playing":
                        # Clear canvas
                        self.canvas.fill(self.WHITE)
                        self.prediction = None
                        self.confidence = 0
                    
                    elif event.key == pygame.K_d:
                        # Toggle debug mode
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                
                # Handle mouse events for drawing
                self.handle_mouse_events(event)
            
            # Draw appropriate screen
            if self.game_state == "menu":
                self.draw_menu()
            elif self.game_state == "playing":
                self.draw_game()
                # Check win condition continuously
                if self.check_win_condition():
                    self.game_state = "gameover"
            elif self.game_state == "gameover":
                self.draw_gameover()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    game = QuickDrawGame()
    game.run()
