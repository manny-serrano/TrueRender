# <img width="200" height="200" alt="Image" src="https://github.com/user-attachments/assets/2ce13084-fb7f-4dbb-b635-8e5bc0274ed3" /> TrueRender

TrueRender is an end-to-end pipeline that reconstructs any real object from casual phone video into a physically accurate 3D digital asset with true geometry, material properties, and elasticity estimated entirely by ML. The resulting asset is simultaneously exportable as a print-ready STL file and deployable as a live AR experience where you can place, touch, and physically interact with the object using your bare hands.



## System Overview

![Image](https://github.com/user-attachments/assets/f4dce161-1d19-45f5-a4dc-2fd7c3a762ac)


## Models Used
- COLMAP
- SAM 3 META
- 3D Gaussian Splatting 
- Neural-PBIR 
- PhysDreamer 
- Sonata + DINOv2


## Current Status

Full pipeline from video to fram extraction -> COLMAP -> SAM 3 META -> 3D Gaussian Splatting 

