import pygame
import time
from .screen_element import *
from .screen_label import *

class ScreenTextbox(ScreenElement):
    
    def __init__(self, name, text, text_size, width, height = 0, text_color = None, back_color = None ):
        ScreenElement.__init__(self, name)
        
        self.max_length = max(100, len(text))
        
        self.padding = (10, 15, 10, 15)
        self.text_size = text_size
        self.width = width
        self.height = height
        self.original_height = height
        
        #check colors....
        #...text...
        if text_color == None:
            self.text_color = (0,0,0)
        else:
            self.text_color = text_color
        #... background...
        if back_color == None:
            self.back_color = (255, 255, 255)
        else:
            self.back_color = back_color
        
        self.is_highlighted = False
        
        #create an inner label          
        self.updateText( text )
        
        
    def setPadding(self, top, right, bottom, left):
        self.padding = (top, right, bottom, left)
        
        self.updateText(self.text)
        
    def set_colors(self, text_color, back_color ):
        self.text_color = text_color
        self.back_color = back_color
            
        self.updateText( self.text )
    
    
    def updateText(self, new_text):
        #check text length ...
        if len(new_text) > self.max_length:
            new_text = new_text[:self.max_length]
            
        self.text = new_text
        
        #calculate max inner width for text...
        # width - padding_right - padding_left
        max_inner_width = self.width - (self.padding[1] + self.padding[3])
        
        #now, create an inner label
        self.inner_label = ScreenLabel(self.name + "__LABEL__", self.text, self.text_size, max_inner_width, 0)
        self.inner_label.position = (self.padding[3], self.padding[0])
        self.inner_label.set_color( self.text_color )
        self.inner_label.set_background( self.back_color )
        
        base_height = self.inner_label.height
        
        self.height = self.original_height
        min_height = self.padding[0] + self.padding[2] + self.inner_label.height
        if min_height > self.height:
            self.height = min_height
        
        #now, the label should be rendered on the background...
        self.rendered = pygame.Surface( (self.width, self.height ) )
        self.rendered.fill( self.back_color )
        
        #create the normal view...
        self.inner_label.render( self.rendered )
        
        #now, create the highlighted view without caret...
        self.inner_label.set_color( self.back_color )
        self.inner_label.set_background( self.text_color )
        
        self.highlighted_nocar = pygame.Surface( (self.width, self.height ) )
        self.highlighted_nocar.fill( self.text_color )
        
        #render highlighted without caret...
        self.inner_label.render( self.highlighted_nocar )
        
        #finally, create the highlighted view with caret...        
        self.inner_label.set_text( self.text + '|' )
        caret_height = self.inner_label.height
        
        self.highlighted_car = pygame.Surface( (self.width, self.height ) )
        self.highlighted_car.fill( self.text_color )
        
        #render highlighted without caret...
        self.inner_label.render( self.highlighted_car )
        
        self.draw_caret = (base_height == caret_height) 
        
        
    def render(self, background, off_x = 0, off_y = 0):
        #for border...
        border_w = 2
        rect_x = self.position[0] + off_x
        rect_y = self.position[1] + off_y
        rect_w = self.width
        rect_h = self.height
        rect = ( rect_x, rect_y, rect_w, rect_h ) 
        
        #draw text on the background texture...
        if not self.is_highlighted:            
            #normal view            
            background.blit( self.rendered, (self.position[0] + off_x, self.position[1] + off_y ) )
            
            pygame.draw.rect( background, self.text_color, rect, border_w)
        else:
            #highlighted view...
            
            current_time = time.time()
            decimal = current_time - int(current_time)
            
            if int((decimal * 100) / 25) % 2 == 0 or not self.draw_caret:            
                #without caret
                background.blit( self.highlighted_nocar, (self.position[0] + off_x, self.position[1] + off_y ) )
            else:
                #with caret
                background.blit( self.highlighted_car, (self.position[0] + off_x, self.position[1] + off_y ) )
                        
            pygame.draw.rect( background, self.back_color, rect, border_w)
    
    def on_mouse_button_click(self, pos, button):
        #get focus...
        if button == 1:
            self.set_focus()
            
            
    def set_focus(self):
        #request focus from parent...
        got_focus = self.parent.set_text_focus( self )
        #set focus....
        self.is_highlighted = got_focus
            
    def lost_focus(self):
        #the writing focus has been lost!
        self.is_highlighted = False
            
    
    
    def on_key_up(self, scancode, key, unicode):
        if self.is_highlighted:
            if len(unicode) > 0:
                if key == 27:
                    #print "ESCAPE!"
                    pass
                elif key == 13:
                    #print "ENTER!"
                    pass
                elif key == 8:
                    #print "BACKSPACE!"
                    if len(self.text) >= 1:
                        self.updateText(self.text[:-1])
                elif key == 9:
                    pass
                    #print "TAB!"
                else:
                    self.updateText(self.text + unicode) 
            else:
                #print "CONTROL KEY!"
                pass