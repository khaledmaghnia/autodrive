import numpy as np
import cv2

class LaneDetector():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None

    def __extract_lanes(self, warped, verbose=False):
        hist = np.sum(warped[350:], axis=0)
        midpoint = warped.shape[1]//2

        left_start = np.argmax(hist[:midpoint])
        right_start = np.argmax(hist[midpoint:])+midpoint

        nonzero = warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])


        self.margin = 100
        n_windows = 9
        window_height = int(warped.shape[0]/n_windows)
        min_pix = 50

        self.left_lane_inds = []
        self.right_lane_inds = []


        for i in range(n_windows):
            left_bottomleft_x = left_start - self.margin
            left_bottomleft_y = warped.shape[0] - i * window_height

            left_topright_x = left_start + self.margin
            left_topright_y = left_bottomleft_y - window_height

            right_bottomleft_x = right_start - self.margin
            right_bottomleft_y = warped.shape[0] - i * window_height

            right_topright_x = right_start + self.margin
            right_topright_y = right_bottomleft_y - window_height
            
            if verbose:
                cv2.rectangle(self.out_img, (left_bottomleft_x, left_topright_y), (left_topright_x, left_bottomleft_y),(0,255,0), thickness=2)
                cv2.rectangle(self.out_img, (right_bottomleft_x, right_topright_y), (right_topright_x, right_bottomleft_y),(0,255,0), thickness=2)


            left_nonzero = (((self.nonzerox >= left_bottomleft_x) & (self.nonzerox <= left_topright_x)) & ((self.nonzeroy <= left_bottomleft_y) & (self.nonzeroy >= left_topright_y))).nonzero()[0]
            right_nonzero = (((self.nonzerox >= right_bottomleft_x) & (self.nonzerox <= right_topright_x)) & ((self.nonzeroy <= right_bottomleft_y) & (self.nonzeroy >= right_topright_y))).nonzero()[0]

            self.left_lane_inds.append(left_nonzero)
            self.right_lane_inds.append(right_nonzero)
            
            if len(left_nonzero) > min_pix:
                left_start = np.int(np.mean(self.nonzerox[left_nonzero]))
            
            if len(right_nonzero) > min_pix:
                right_start = np.int(np.mean(self.nonzerox[right_nonzero]))
            
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)
        
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]

        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]
        
        if verbose:
            self.out_img[self.lefty, self.leftx] = [255, 0, 0]
            self.out_img[self.righty, self.rightx] = [0, 0, 255]

    def fit_lanes(self, warped, left_fit, right_fit, p_left_fitx, p_right_fitx, verbose=False):
        #for visualization
        self.out_img = np.dstack((warped, warped, warped))

        if np.array(left_fit).any() != None:
            found = self.search_around_poly(warped, left_fit, right_fit)
            if found == False:
                self.__extract_lanes(warped, verbose=verbose)
        else:
            self.__extract_lanes(warped, verbose=verbose)

        if 0 in [len(self.leftx), len(self.lefty), len(self.rightx), len(self.righty)]:
            self.left_fitx, self.right_fitx = p_left_fitx, p_right_fitx
            self.ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        else:
            self.left_fitx, self.right_fitx, self.ploty = self.fit_poly(warped.shape, self.leftx, self.lefty, self.rightx, self.righty)
        
        # print(self.left_fitx.shape)
        # print(self.ploty.shape)
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        self.out_img = cv2.fillPoly(self.out_img, np.int_([pts]), (0,255, 0))

        return self.out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
         ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        try:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        except TypeError as e:
            print("error")
            pass

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using self.ploty, left_fit and right_fit ###
        try:
            left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*self.ploty**2 + 1*self.ploty
            right_fitx = 1*self.ploty**2 + 1*self.ploty
        
        return left_fitx, right_fitx, self.ploty

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        self.margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        
       
        self.left_lane_inds = ((self.nonzerox > (left_fit[0]*(self.nonzeroy**2) + left_fit[1]*self.nonzeroy + 
                left_fit[2] - self.margin)) & (self.nonzerox < (left_fit[0]*(self.nonzeroy**2) + 
                left_fit[1]*self.nonzeroy + left_fit[2] + self.margin)))
        self.right_lane_inds = ((self.nonzerox > (right_fit[0]*(self.nonzeroy**2) + right_fit[1]*self.nonzeroy + 
                right_fit[2] - self.margin)) & (self.nonzerox < (right_fit[0]*(self.nonzeroy**2) + 
                right_fit[1]*self.nonzeroy + right_fit[2] + self.margin)))
        
        # Again, extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        if 0 in [len(self.leftx), len(self.lefty), len(self.rightx), len(self.righty)]:
            found = False
        else:
            found = True
        return found

    def measure_curvature_pixels(self, left_fit_cr, right_fit_cr):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''
        ym_per_pix = 30/500 # meters per pixel in y dimension
        xm_per_pix = 3.7/800 # meters per pixel in x dimension
        
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return left_curverad, right_curverad