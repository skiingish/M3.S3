//ToDo: Add Comment (what is the purpose of this function? Where its going to get executed?)
__kernel void square_magnitude(const int size,
                      __global int* v, __global int* v2, __global int* v3) {
    
    // Thread identifiers
    const int globalIndex = get_global_id(0);
    
    //printf("Item in Array 1 :(%d)\n ", v[globalIndex]);
    //printf("Item in Array 2 :(%d)\n ", v2[globalIndex]);
    
    //uncomment to see the index each PE works on
    
    //printf("Kernel process index :(%d)\n ", globalIndex);
    
    v3[globalIndex] = v[globalIndex] + v2[globalIndex];
    
    //v[globalIndex] = v[globalIndex] * v[globalIndex];
    
    //printf("Item in Result :(%d)\n ", v3[globalIndex]);
    
    //printf("Kernel process index :(%d)\n ", globalIndex);
}
