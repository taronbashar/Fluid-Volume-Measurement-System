from fvms.drivers import Lcd
# from time import sleep

_ldc = Lcd()

def display(s):
    lines = s.split('\n')
    if len(lines) > 2:
        raise ValueError("Display can handle at most 2 lines")
    
    _ldc.lcd_clear()
    if len(lines) == 1:
        _ldc.lcd_display_string(lines[0], 1)
    elif len(lines) == 2:
        _ldc.lcd_display_string(lines[0], 1)
        _ldc.lcd_display_string(lines[1], 2)

# if __name__ == "__main__":
#     # Load the driver and set it to "display"
#     # If you use something from the driver library use the "display." prefix first
#     display = drivers.Lcd()

#     # Main body of code
#     try:
#         while True:
#             # Remember that your sentences can only be 16 characters long!
#             print("Writing to display")
#             display.lcd_display_string("Greetings Human!", 1)  # Write line of text to first line of display
#             display.lcd_display_string("Demo Pi Guy code", 2)  # Write line of text to second line of display
#             sleep(2)                                           # Give time for the message to be read
#             display.lcd_display_string("I am a display!", 1)   # Refresh the first line of display with a different message
#             sleep(2)                                           # Give time for the message to be read
#             display.lcd_clear()                                # Clear the display of any data
#             sleep(2)                                           # Give time for the message to be read
#     except KeyboardInterrupt:
#         # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
#         print("Cleaning up!")
#         display.lcd_clear()
    