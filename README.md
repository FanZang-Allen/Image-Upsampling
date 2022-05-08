# Image-Upsampling

To run the code on an image (png or jpg) using default 1 iteration feedback loop, run the following command:
```
python upsample.py -p 'path-to-file'
```

To run the code using default "n" iterations of feedback loop, run the following command by passing an integer after -l:
```
python upsample.py -p 'path-to-file' -l n
```

Output: 
```
Upsampling chip_input.png using 2 feedback loops...
one iteration finished
one iteration finished
Upsampled image is saved at:  path-to-new-file
```
