CC = g++
LD = -L/usr/lib/x86_64-linux-gnu -lSDL2 -lGLEW -lGL
INC = -I/usr/include -D_REENTRANT

LD2 = -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d
INC2 = -I/usr/include -D_opencv2 -D_opencv

pc_viewer: pc_viewer.o
	$(CC) -o pc_viewer pc_viewer.o $(LD) $(INC) $(LD2) $(INC2)

pc_viewer.o: pc_viewer.cpp
	$(CC) -std=c++11 -c -o pc_viewer.o pc_viewer.cpp $(LD) $(INC) $(LD2) $(INC2)

clean:
	rm pc_viewer.o pc_viewer
