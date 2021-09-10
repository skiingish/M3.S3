// It runs within the kernel that was created, it takes the size of the data,
// and the array of vectors.
// it then squares the vector and replaces the value in the array.
// It gets executed from the main loop within the main function when this program kernel get’s added to the command queue. 

// EDIT: changed to take in the 3 different arrays, 2 to be added together and 1 to store the result.
__kernel void square_magnitude(const int size,
                      __global int* v, __global int* v2, __global int* v3) {
    
    // Thread identifiers
    const int globalIndex = get_global_id(0);
    
    //printf("Item in Array 1 :(%d)\n ", v[globalIndex]);
    //printf("Item in Array 2 :(%d)\n ", v2[globalIndex]);
    
    //uncomment to see the index each PE works on
    
    //printf("Kernel process index :(%d)\n ", globalIndex);
    
    v3[globalIndex] = v[globalIndex] + v2[globalIndex];
    
    // Old Square Function.
    //v[globalIndex] = v[globalIndex] * v[globalIndex];
    
    //printf("Item in Result :(%d)\n ", v3[globalIndex]);
}
