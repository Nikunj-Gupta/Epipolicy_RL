#include <stdint.h>

void load(uint64_t* p, uint64_t* ret){
  __atomic_load(p, ret, __ATOMIC_SEQ_CST);
}

void store(uint64_t* p, uint64_t* val){
  __atomic_store(p, val, __ATOMIC_SEQ_CST);
}

uint64_t fetch_add(uint64_t* p, uint64_t* pval){
  return __atomic_fetch_add(p, *pval, __ATOMIC_SEQ_CST);
}

uint64_t fetch_sub(uint64_t* p, uint64_t* pval){
  return __atomic_fetch_sub(p, *pval, __ATOMIC_SEQ_CST);
}
