#include <stdint.h>

void load(int64_t* p, int64_t* ret){
  __atomic_load(p, ret, __ATOMIC_SEQ_CST);
}

void store(int64_t* p, int64_t* val){
  __atomic_store(p, val, __ATOMIC_SEQ_CST);
}

int64_t fetch_add(int64_t* p, int64_t* pval){
  return __atomic_fetch_add(p, *pval, __ATOMIC_SEQ_CST);
}

int64_t fetch_sub(int64_t* p, int64_t* pval){
  return __atomic_fetch_sub(p, *pval, __ATOMIC_SEQ_CST);
}
