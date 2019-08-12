
all:

	g++ -ggdb `pkg-config opencv --cflags` trackers.cpp -o `basename trackers.cpp .cpp` `pkg-config opencv --libs` -lX11 -lXrandr -lXinerama -lXi -lXxf86vm -lXcursor -ldl -lm
	
clean:

	rm trackers.o
