"""
Global Component Remover for fNIRS Data (Fixed Version)
======================================================

Author: Xian Zhang, PhD (Dr. Zhang, Xian)
Affiliation: Brain Function Lab, Department of Psychiatry,
             Yale School of Medicine
Contact: xzhang63@gmail.com

Description:
    Implements spatial kernel-weighted estimation and removal of a global
    component from fNIRS time series based on angular distances
    between channel positions on the scalp. Channel neighborhoods are
    formed using a Gaussian kernel in spherical (TH, PHI) space.

    This version uses a simplified distance calculation method that matches
    the updated MATLAB implementation exactly.

References (please cite when using this code):
    - "Separation of the global and local components in functional
       near-infrared spectroscopy signals using principal component
       spatial filtering"
    - "Signal processing of functional NIRS data acquired during overt
       speaking" (demonstrates global component in deoxy-Hb)

Usage:
    from global_remover_fixed import GlobalRemover
    gr = GlobalRemover(xyz, sigma_degrees)
    v_global = gr.get_global(v_raw)
    v_clean = gr.remove(v_raw)

Inputs:
    xyz            - Array of shape (n_channels, 3) with Cartesian coordinates (x,y,z)
    sigma_degrees  - Gaussian kernel width in degrees (scalar)

Notes:
    - Requires numpy and scipy for distance calculations
    - Coordinates should be Cartesian (x,y,z) on the scalp surface
    - Kernel weights are column-normalized to sum to 1
    - Sigma should typically be between 20-50 degrees
"""

import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform


class GlobalRemover:
    """
    Global component removal for fNIRS data using spatial kernel weighting.
    
    This class implements a method to estimate and remove global components
    from fNIRS channel time series by computing spatial kernels based on
    angular distances between channel positions on the scalp.
    """
    
    def __init__(self, xyz, sigma):
        """
        Initialize the GlobalRemover with channel coordinates and kernel width.
        
        Parameters:
        -----------
        xyz : array_like, shape (n_channels, 3)
            Cartesian coordinates (x, y, z) of fNIRS channels
        sigma : float
            Gaussian kernel width in degrees
        """
        self.sigma = sigma
        self.xyz = np.array(xyz)
        self.n_ch = self.xyz.shape[0]
        
        # Validate sigma range
        if self.sigma < 20:
            warnings.warn('Sigma should be between 20-50 degrees', 
                        UserWarning)
        
        # Convert to spherical coordinates for angular distance calculation
        x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
        theta, phi, _ = self._cart2sph(x, y, z)
        
        # Calculate distance matrix using the simplified method from MATLAB
        self.distance_matrix = self._calculate_distance_matrix(theta, phi)
        
        # Create Gaussian kernel in angular space (sigma in degrees)
        sigma_rad = self.sigma * np.pi / 180
        kernel = np.exp(-self.distance_matrix**2 / (2 * sigma_rad**2))
        
        # Column-normalize kernel so weights sum to 1 for each target channel
        self.kernel = np.zeros_like(kernel)
        for i in range(self.n_ch):
            col_sum = np.sum(kernel[:, i])
            if col_sum == 0:
                self.kernel[:, i] = kernel[:, i]
            else:
                self.kernel[:, i] = kernel[:, i] / col_sum
    
    def _cart2sph(self, x, y, z):
        """
        Convert Cartesian coordinates to spherical coordinates.
        This function replicates MATLAB's cart2sph behavior exactly.
        
        MATLAB cart2sph returns [azimuth, elevation, r] where:
        - azimuth: angle in xy-plane from positive x-axis (0 to 2π)
        - elevation: angle from xy-plane to point (-π/2 to π/2)
        - r: distance from origin
        
        Parameters:
        -----------
        x, y, z : array_like
            Cartesian coordinates
            
        Returns:
        --------
        azimuth, elevation, r : ndarray
            Spherical coordinates matching MATLAB's cart2sph output
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Azimuth: angle in xy-plane from positive x-axis
        azimuth = np.arctan2(y, x)
        
        # Elevation: angle from xy-plane to point
        # This is different from colatitude!
        elevation = np.arcsin(z / r)
        
        return azimuth, elevation, r
    
    def _calculate_distance_matrix(self, theta, phi):
        """
        Calculate distance matrix using the simplified method from MATLAB.
        
        This method replicates the MATLAB code exactly:
        1. Calculate differences in theta and phi
        2. Handle circular nature of angles
        3. Use Euclidean distance in angular space
        
        Parameters:
        -----------
        theta, phi : ndarray
            Spherical coordinates (azimuth, elevation)
            
        Returns:
        --------
        ndarray, shape (n_ch, n_ch)
            Distance matrix
        """
        n_ch = len(theta)
        
        # Initialize difference arrays
        diffv = np.zeros((n_ch, n_ch, 2))
        
        # Calculate differences for all pairs
        for i in range(n_ch):
            for j in range(i, n_ch):
                diffv[i, j, 0] = theta[i] - theta[j]  # theta difference
                diffv[i, j, 1] = phi[i] - phi[j]      # phi difference
                diffv[j, i, :] = diffv[i, j, :]       # symmetric
        
        # Handle circular nature of angles
        diffv2 = diffv.copy()
        for i in range(2):
            temp = diffv[:, :, i]
            # Handle theta wrapping: -π to π
            temp[temp >= np.pi] = temp[temp >= np.pi] - 2 * np.pi
            temp[temp <= -np.pi] = temp[temp <= -np.pi] + 2 * np.pi
            diffv2[:, :, i] = temp
        
        # Calculate Euclidean distance in angular space
        distance_matrix = np.sqrt(diffv2[:, :, 0]**2 + diffv2[:, :, 1]**2)
        
        return distance_matrix
    
    def get_global(self, v_raw):
        """
        Estimate the global component for each channel as a
        kernel-weighted spatial average of signals across channels.
        
        Parameters:
        -----------
        v_raw : array_like, shape (n_channels, n_timepoints)
            Raw fNIRS time series data
            
        Returns:
        --------
        ndarray, shape (n_channels, n_timepoints)
            Estimated global component for each channel
        """
        v_raw = np.array(v_raw)
        if v_raw.ndim == 1:
            v_raw = v_raw.reshape(-1, 1)
        
        # Ensure we only use the first n_ch channels
        v_raw = v_raw[:self.n_ch, :]
        
        # Compute global component as kernel-weighted average
        v_global = (v_raw.T @ self.kernel).T
        
        return v_global
    
    def remove(self, v_raw):
        """
        Remove the estimated global component from the raw signal.
        
        Parameters:
        -----------
        v_raw : array_like, shape (n_channels, n_timepoints)
            Raw fNIRS time series data
            
        Returns:
        --------
        ndarray, shape (n_channels, n_timepoints)
            Cleaned data with global component removed
        """
        v_raw = np.array(v_raw)
        v_global = self.get_global(v_raw)
        v_clean = v_raw - v_global
        return v_clean
    
    def save_to_mat(self, filename='global_remover_data.mat'):
        """
        Save thphi coordinates and distance matrix to a MATLAB .mat file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the .mat file to save (default: 'global_remover_data.mat')
        """
        try:
            from scipy.io import savemat
            
            # Convert to spherical coordinates for saving
            x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
            theta, phi, _ = self._cart2sph(x, y, z)
            thphi = np.column_stack([theta, phi])
            
            # Prepare data for saving
            mat_data = {
                'thphi': thphi,
                'distance_matrix': self.distance_matrix,
                'xyz': self.xyz,
                'sigma': self.sigma,
                'n_ch': self.n_ch,
                'kernel': self.kernel
            }
            
            # Save to .mat file
            savemat(filename, mat_data)
            print(f"Data saved to {filename}")
            print(f"  - thphi shape: {thphi.shape}")
            print(f"  - distance_matrix shape: {self.distance_matrix.shape}")
            print(f"  - sigma: {self.sigma} degrees")
            print(f"  - n_ch: {self.n_ch}")
            
        except ImportError:
            print("scipy.io not available. Cannot save .mat files.")
            print("Please install scipy: pip install scipy")
        except Exception as e:
            print(f"Error saving to .mat file: {e}")
    
    @staticmethod
    def demo():
        """
        Demonstration function showing how to use GlobalRemover.
        Requires testdata.mat to be available.
        """
        try:
            from scipy.io import loadmat
            
            # Load test data
            data = loadmat('testdata.mat')
            xyz = data['meanxyz']
            meandata = data['meandata']
            sigma = 46
            
            print(f"Data shape: {meandata.shape}")
            print("Note: We encourage concatenating all data because global component")
            print("should be shared across all conditions and runs")
            
            # Reshape data: (n_data, n_ch, n_condition_or_run) -> (n_ch, n_data*n_condition, n_signal)
            n_signal = meandata.shape[2]  # 2 for oxy and deoxy
            data0 = np.transpose(meandata, (1, 0, 2))  # (n_ch, n_data, n_signal)
            data = data0.reshape(data0.shape[0], -1)  # (n_ch, n_data*n_condition)
            sz = data.shape
            
            # Define bad channels
            bad_channel = [0]  # 0-based indexing for Python
            ind = list(range(data.shape[0]))
            for ch in bad_channel:
                if ch in ind:
                    ind.remove(ch)
            
            # Initialize GlobalRemover
            gr = GlobalRemover(xyz[ind, :], sigma)
            
            # Save thphi and distance matrix to .mat file
            gr.save_to_mat('global_remover_analysis.mat')
            
            # Process data
            global_c = np.zeros_like(data)
            clean_data = np.zeros_like(data)
            
            global_c[ind, :] = gr.get_global(data[ind, :])
            clean_data[ind, :] = gr.remove(data[ind, :])
            
            # Reshape back to original format
            clean_data = clean_data.reshape(sz[0], sz[1] // n_signal, n_signal)
            global_c = global_c.reshape(sz[0], sz[1] // n_signal, n_signal)
            
            # Save clean_data and other results for MATLAB comparison
            save_data = {
                'cleanData': clean_data,
                'globalC': global_c,
                'data0': data0,
                'kernel': gr.kernel,
                'distance_matrix': gr.distance_matrix,
                'sigma': sigma,
                'nSignal': n_signal
            }
            
            # Save to .mat file for MATLAB comparison
            from scipy.io import savemat
            savemat('python_results.mat', save_data)
            print("Python results saved to 'python_results.mat' for MATLAB comparison")
            print(f"  - cleanData shape: {clean_data.shape}")
            print(f"  - globalC shape: {global_c.shape}")
            print(f"  - data0 shape: {data0.shape}")
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original data correlation (match MATLAB: corrcoef(data0(:,:,1)'))
            corr_orig = np.corrcoef(data0[:, :, 0])
            im1 = axes[0, 0].imshow(corr_orig, vmin=-1, vmax=1, cmap='RdBu_r')
            axes[0, 0].set_title('Data, Oxy')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Cleaned data correlation (match MATLAB: corrcoef(cleanData(:,:,1)'))
            corr_clean = np.corrcoef(clean_data[:, :, 0])
            im2 = axes[0, 1].imshow(corr_clean, vmin=-1, vmax=1, cmap='RdBu_r')
            axes[0, 1].set_title('Cleaned Data, Oxy')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Global component correlation (match MATLAB: corrcoef(globalC(:,:,1)'))
            corr_global = np.corrcoef(global_c[:, :, 0])
            im3 = axes[1, 0].imshow(corr_global, vmin=-1, vmax=1, cmap='RdBu_r')
            axes[1, 0].set_title('Global Component, Oxy')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Kernel visualization
            im4 = axes[1, 1].imshow(gr.kernel, cmap='viridis')
            axes[1, 1].set_title('Convolution Kernel')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig('globalremovaldemo.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Reshape clean_data back to original format
            clean_data = np.transpose(clean_data, (1, 0, 2))
            
            print("Demo completed successfully!")
            print(f"Clean data shape: {clean_data.shape}")
            
        except ImportError:
            print("scipy.io not available. Cannot load .mat files.")
            print("Please install scipy: pip install scipy")
        except FileNotFoundError:
            print("testdata.mat not found. Please ensure the file is in the current directory.")
        except Exception as e:
            print(f"Error running demo: {e}")


if __name__ == "__main__":
    # Run demo if script is executed directly
    GlobalRemover.demo()
    
    # Example of saving data to .mat file
    print("\n" + "="*50)
    print("Example: Saving thphi and distance matrix to .mat file")
    print("="*50)
    
    # Create example data
    import numpy as np
    xyz_example = np.random.randn(5, 3)  # 5 channels with 3D coordinates
    sigma_example = 46
    
    # Initialize GlobalRemover
    gr_example = GlobalRemover(xyz_example, sigma_example)
    
    # Save to .mat file
    gr_example.save_to_mat('example_global_remover.mat')
    
    print("\nYou can now load this data in MATLAB using:")
    print("load('example_global_remover.mat')")
    print("The variables 'thphi' and 'distance_matrix' will be available.")

