// TODO: Implement fast stack allocation.
#ifndef _SCTL_MEM_MGR_HPP_
#define _SCTL_MEM_MGR_HPP_

//#include <mutex>

#include <sctl/common.hpp>

#include <omp.h>
#include <typeinfo>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <stack>
#include <map>
#include <set>

namespace SCTL_NAMESPACE {

#ifdef SCTL_MEMDEBUG

template <class ValueType> class ConstIterator {

  template <typename T> friend class ConstIterator;

  template <typename T> friend class Iterator;

  void IteratorAssertChecks(Long j = 0) const;

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef const ValueType* pointer;
  typedef const ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 protected:
  char* base;
  difference_type len, offset;
  Long alloc_ctr;
  void* mem_head;
  static const Long ValueSize = sizeof(ValueType);

 public:
  ConstIterator();

  explicit ConstIterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  template <class AnotherType> explicit ConstIterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  reference operator*() const;

  pointer operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  ConstIterator& operator++();

  ConstIterator operator++(int);

  ConstIterator& operator--();

  ConstIterator operator--(int);

  // Arithmetic
  ConstIterator& operator+=(difference_type i);

  ConstIterator operator+(difference_type i) const;

  template <class T> friend ConstIterator<T> operator+(typename ConstIterator<T>::difference_type i, const ConstIterator<T>& right);

  ConstIterator& operator-=(difference_type i);

  ConstIterator operator-(difference_type i) const;

  difference_type operator-(const ConstIterator& I) const;

  // Comparison operators
  bool operator==(const ConstIterator& I) const;

  bool operator!=(const ConstIterator& I) const;

  bool operator<(const ConstIterator& I) const;

  bool operator<=(const ConstIterator& I) const;

  bool operator>(const ConstIterator& I) const;

  bool operator>=(const ConstIterator& I) const;

  friend std::ostream& operator<<(std::ostream& out, const ConstIterator& I) {
    out << "(" << (long long)I.base << "+" << I.offset << ":" << I.len << ")";
    return out;
  }
};

template <class ValueType> class Iterator : public ConstIterator<ValueType> {

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef ValueType* pointer;
  typedef ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 public:
  Iterator();

  explicit Iterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  template <class AnotherType> explicit Iterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  reference operator*() const;

  value_type* operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  Iterator& operator++();

  Iterator operator++(int);

  Iterator& operator--();

  Iterator operator--(int);

  // Arithmetic
  Iterator& operator+=(difference_type i);

  Iterator operator+(difference_type i) const;

  template <class T> friend Iterator<T> operator+(typename Iterator<T>::difference_type i, const Iterator<T>& right);

  Iterator& operator-=(difference_type i);

  Iterator operator-(difference_type i) const;

  difference_type operator-(const ConstIterator<ValueType>& I) const;
};

template <class ValueType, Long DIM> class StaticArray {
  typedef Long difference_type;

 public:
  StaticArray() = default;

  StaticArray(const StaticArray&) = default;

  StaticArray& operator=(const StaticArray&) = default;

  explicit StaticArray(std::initializer_list<ValueType> arr_);

  ~StaticArray() = default;

  // value_type* like operators
  const ValueType& operator*() const;

  ValueType& operator*();

  const ValueType* operator->() const;

  ValueType* operator->();

  const ValueType& operator[](difference_type off) const;

  ValueType& operator[](difference_type off);

  operator ConstIterator<ValueType>() const;

  operator Iterator<ValueType>();

  // Arithmetic
  ConstIterator<ValueType> operator+(difference_type i) const;

  Iterator<ValueType> operator+(difference_type i);

  template <class T, Long d> friend ConstIterator<T> operator+(typename StaticArray<T,d>::difference_type i, const StaticArray<T,d>& right);

  template <class T, Long d> friend Iterator<T> operator+(typename StaticArray<T,d>::difference_type i, StaticArray<T,d>& right);

  ConstIterator<ValueType> operator-(difference_type i) const;

  Iterator<ValueType> operator-(difference_type i);

  difference_type operator-(const ConstIterator<ValueType>& I) const;

  // Comparison operators
  bool operator==(const ConstIterator<ValueType>& I) const;

  bool operator!=(const ConstIterator<ValueType>& I) const;

  bool operator< (const ConstIterator<ValueType>& I) const;

  bool operator<=(const ConstIterator<ValueType>& I) const;

  bool operator> (const ConstIterator<ValueType>& I) const;

  bool operator>=(const ConstIterator<ValueType>& I) const;

 private:

  ValueType arr_[DIM];
};

#endif

template <class ValueType> Iterator<ValueType> NullIterator();
template <class ValueType> Iterator<ValueType> Ptr2Itr(void* ptr, Long len);
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len);

/**
 * \brief MemoryManager class declaration.
 */
class MemoryManager {

 public:
  static constexpr char init_mem_val = 42;
  static constexpr Long end_padding = 64;

  /**
   * \brief Header data for each memory block.
   */
  struct MemHead {
    typedef decltype(typeid(char).hash_code()) TypeID;
    Long n_indx;
    Long n_elem;
    Long type_size;
    Long alloc_ctr;
    TypeID type_id;
    #ifdef SCTL_MEMDEBUG
    unsigned char check_sum;
    #endif
  };

  /**
   * \brief Constructor for MemoryManager.
   */
  explicit MemoryManager(Long N);

  /**
   * \brief Constructor for MemoryManager.
   */
  ~MemoryManager();

  static MemHead& GetMemHead(char* p);

  static void CheckMemHead(const MemHead& p);

  Iterator<char> malloc(const Long n_elem, const Long type_size = sizeof(char), const MemHead::TypeID type_id = typeid(char).hash_code()) const;

  void free(Iterator<char> p) const;

  void print() const;

  static void test();

  // Check all free memory equals init_mem_val
  void Check() const;

  // A global MemoryManager object. This is the default for aligned_new and aligned_free
  static MemoryManager& glbMemMgr() {
    static MemoryManager m(SCTL_GLOBAL_MEM_BUFF * 1024LL * 1024LL);
    return m;
  }

 private:
  // Private constructor
  MemoryManager();

  // Private copy constructor
  MemoryManager(const MemoryManager& m);

  /**
   * \brief Node structure for a doubly linked list, representing free and
   * occupied memory blocks. Blocks are split, merged or state is changed
   * between free and occupied in O(1) time given the pointer to the MemNode.
   */
  struct MemNode {
    bool free;
    Long size;
    char* mem_ptr;
    Long prev, next;
    std::multimap<Long, Long>::iterator it;
  };

  /**
   * \brief Return index of one of the available MemNodes from node_stack or
   * create new MemNode by resizing node_buff.
   */
  Long new_node() const;

  /**
   * \brief Add node index for now available MemNode to node_stack.
   */
  void delete_node(Long indx) const;

  char* buff;         // pointer to memory buffer.
  Long buff_size;     // total buffer size in bytes.
  Long n_dummy_indx;  // index of first (dummy) MemNode in link list.

  mutable std::vector<MemNode> node_buff;      // storage for MemNode objects, this can only grow.
  mutable std::stack<Long> node_stack;         // stack of available free MemNodes from node_buff.
  mutable std::multimap<Long, Long> free_map;  // pair (MemNode.size, MemNode_id) for all free MemNodes.
  //mutable omp_lock_t omp_lock;                 // openmp lock to prevent concurrent changes.
  //mutable std::mutex mutex_lock;
  mutable std::set<void*> system_malloc;       // track pointers allocated using system malloc.
};

/**
 * \brief Aligned allocation as an alternative to new. Uses placement new to
 * construct objects.
 */
template <class ValueType> Iterator<ValueType> aligned_new(Long n_elem = 1, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

/**
 * \brief Aligned de-allocation as an alternative to delete. Calls the object
 * destructor.
 */
template <class ValueType> void aligned_delete(Iterator<ValueType> A, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

/**
 * \brief Wrapper to memcpy. Also checks if source and destination pointers are
 * the same.
 */
template <class ValueType> Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num);

template <class ValueType> Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num);

}  // end namespace SCTL_NAMESPACE

#include SCTL_INCLUDE(mem_mgr.txx)

#endif  //_SCTL_MEM_MGR_HPP_
