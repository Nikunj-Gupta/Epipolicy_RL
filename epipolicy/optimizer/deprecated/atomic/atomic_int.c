#include <stdint.h>

void load(int32_t* p, int32_t* ret){
  __atomic_load(p, ret, __ATOMIC_SEQ_CST);
}

void store(int32_t* p, int32_t* val){
  __atomic_store(p, val, __ATOMIC_SEQ_CST);
}

int32_t fetch_add(int32_t* p, int32_t* pval){
  return __atomic_fetch_add(p, *pval, __ATOMIC_SEQ_CST);
}

int32_t fetch_sub(int32_t* p, int32_t* pval){
  return __atomic_fetch_sub(p, *pval, __ATOMIC_SEQ_CST);
}