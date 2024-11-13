import pygame

# Initialize Pygame for sound
pygame.mixer.init()

# Load the sound
alert_sound = pygame.mixer.Sound("C:/Users/ajkan/OneDrive/Desktop/pm_final/alert.wav")

# Play the sound
alert_sound.play()

# Wait for the sound to finish
pygame.time.delay(2000)  # Adjust the delay if needed to ensure the sound plays
