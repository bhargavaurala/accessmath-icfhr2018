
import pygame
from pygame import Surface, Rect

from .screen_element import ScreenElement

class ScreenCanvasRectangle:
    def __init__(self, x, y, w, h, visible=True):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.visible = visible

class ScreenCanvas(ScreenElement):

    def __init__(self, name, width, height):
        ScreenElement.__init__(self, name)

        self.width = width
        self.height = height

        self.elements = {}
        self.draw_order = []
        self.name_order = {}
        self.selected_element = None

        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        self.min_width = 10
        self.min_height = 10
        self.sel_size = 10
        self.sel_colors = [(128,0,0), (0,128,0), (0,0, 128), (128,128,0), (128,0,128), (0,128,128)]

        self.locked = False
        self.drag_type = -1 # No dragging

        self.object_edited_callback = None
        self.object_selected_callback = None

    def update_names_order(self):
        tempo_names = sorted(self.draw_order)
        self.name_order = {name: idx for idx, name in enumerate(tempo_names)}

    def add_element(self, name, x, y, w, h):
        if name not in self.elements:
            self.elements[name] = ScreenCanvasRectangle(x, y, w, h)
            self.draw_order.insert(0, name)
            self.update_names_order()

    def update_element(self, name, x, y, w, h, visible):
        element = self.elements[name]
        element.x = x
        element.y = y
        element.w = w
        element.h = h
        element.visible = visible

    def rename_element(self, old_name, new_name):
        if old_name in self.elements and old_name != new_name and new_name not in self.elements:
            # old name exists, new name is different and is not in use
            # copy with new name
            self.elements[new_name] = self.elements[old_name]
            # delete old name reference
            del self.elements[old_name]

            # check if selected ....
            if self.selected_element == old_name:
                # update selected name
                self.selected_element = new_name

            pos = self.draw_order.index(old_name)
            self.draw_order.remove(old_name)
            self.draw_order.insert(pos, new_name)

            self.update_names_order()

    def remove_element(self, element_name):
        if element_name in self.elements:
            # delete name reference
            del self.elements[element_name]

            # check if selected ....
            if self.selected_element == element_name:
                # update selected name
                self.selected_element = None

            # no longer drawing
            self.draw_order.remove(element_name)

            self.update_names_order()

    def render(self, background, off_x=0, off_y=0):
        background.set_clip(Rect(self.position[0], self.position[1], self.width, self.height))

        for idx, element in enumerate(self.draw_order):
            # skip elements that are not currently visible
            if not self.elements[element].visible:
                continue

            x = round(self.elements[element].x + self.position[0] + off_x)
            y = round(self.elements[element].y + self.position[1] + off_y)
            w = round(self.elements[element].w)
            h = round(self.elements[element].h)

            name_order_idx = self.name_order[element]

            pygame.draw.rect(background, self.colors[name_order_idx % len(self.colors)], (x, y, w, h), 2)

            if self.selected_element == element:
                # selected
                color = self.sel_colors[name_order_idx % len(self.sel_colors)]
                pygame.draw.rect(background, color, (x - self.sel_size / 2, y - self.sel_size / 2, self.sel_size, self.sel_size))
                pygame.draw.rect(background, color, (x + w - self.sel_size / 2, y - self.sel_size / 2, self.sel_size, self.sel_size))
                pygame.draw.rect(background, color, (x - self.sel_size / 2, y + h - self.sel_size / 2, self.sel_size, self.sel_size))
                pygame.draw.rect(background, color, (x + w - self.sel_size / 2, y + h - self.sel_size / 2, self.sel_size, self.sel_size))

        pygame.draw.rect(background, (0, 0, 0), (self.position[0], self.position[1], self.width, self.height), 2)
        background.set_clip(None)

    def on_mouse_button_down(self, pos, button):
        if self.locked:
            return

        if button == 1:
            pre_selected_element = self.selected_element
            self.selected_element = None

            # Left-click
            px = pos[0]
            py = pos[1]

            half = self.sel_size / 2

            # for every object in reversed draw order ...
            # (things drawn last are visible on top)
            for idx, element in enumerate(reversed(self.draw_order)):
                # skip elements that are not currently visible
                if not self.elements[element].visible:
                    continue

                x = self.elements[element].x + self.position[0]
                y = self.elements[element].y + self.position[1]
                w = self.elements[element].w
                h = self.elements[element].h

                if (x <= px <= x + w) and (y <= py <= y + h):
                    self.drag_type = 0

                if (x - half <= px <= x + half) and (y - half <= py <= y + half):
                    self.drag_type = 1

                if (x + w - half <= px <= x + w + half) and (y - half <= py <= y + half):
                    self.drag_type = 2

                if (x - half <= px <= x + half) and (y + h - half <= py <= y + h + half):
                    self.drag_type = 3

                if (x + w - half <= px <= x + w + half) and (y + h - half <= py <= y + h + half):
                    self.drag_type = 4

                if self.drag_type != -1:
                    self.selected_element = element
                    break

            # if is not top of draw order, move to first ...
            if self.selected_element is not None and self.draw_order[-1] != self.selected_element:
                self.draw_order.remove(self.selected_element)
                #self.draw_order.insert(0, self.selected_element)
                self.draw_order.append(self.selected_element)

            if self.selected_element != pre_selected_element:
                if self.object_selected_callback is not None:
                    self.object_selected_callback(self.selected_element)

    def change_selected_element(self, new_selected_element):
        # only if it is a real change
        if new_selected_element == self.selected_element:
            return

        # copy selected
        self.selected_element = new_selected_element

        # move to last (if needed)
        if self.selected_element is not None and self.draw_order[-1] != self.selected_element:
            self.draw_order.remove(self.selected_element)
            #self.draw_order.insert(0, self.selected_element)
            self.draw_order.append(self.selected_element)

    def on_mouse_button_up(self, pos, button):
        if self.locked:
            return

        self.drag_type = -1

    def on_mouse_enter(self, pos, rel, buttons):
        if self.locked:
            return

        self.drag_type = -1
        #print(pos)

    def on_mouse_leave(self, pos, rel, buttons):
        if self.locked:
            return

        self.drag_type = -1
        #print(pos)

    def on_mouse_motion(self, pos, rel, buttons):
        if self.locked:
            return

        if self.selected_element is None:
            return

        if self.drag_type == -1:
            # not actually dragging/editing anything
            return

        curr_elem = self.elements[self.selected_element]
        dx, dy = rel

        prev_x = curr_elem.x
        prev_y = curr_elem.y
        prev_w = curr_elem.w
        prev_h = curr_elem.h

        if self.drag_type == 0:
            # translation, scale not affected
            curr_elem.x += dx
            curr_elem.y += dy

        if self.drag_type == 1:
            # Top-left corner
            curr_elem.x += dx
            curr_elem.y += dy
            curr_elem.w -= dx
            curr_elem.h -= dy

        if self.drag_type == 2:
            # Top-right
            curr_elem.y += dy
            curr_elem.w += dx
            curr_elem.h -= dy

        if self.drag_type == 3:
            # Bottom-left
            curr_elem.x += dx
            curr_elem.w -= dx
            curr_elem.h += dy

        if self.drag_type == 4:
            # Bottom-right corner
            curr_elem.w += dx
            curr_elem.h += dy

        if curr_elem.w < self.min_width:
            curr_elem.w = self.min_width
        if curr_elem.h < self.min_height:
            curr_elem.h = self.min_height

        if ((prev_x != curr_elem.x or prev_y != curr_elem.y or prev_w != curr_elem.w or prev_h != curr_elem.h) and
            self.object_edited_callback is not None):
            self.object_edited_callback(self, self.selected_element)

        #print(str((pos, rel, buttons)))
        #print(pos)