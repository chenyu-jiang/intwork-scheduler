/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/customer.h"
#include "ps/internal/postoffice.h"
namespace ps {

const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();

Customer::Customer(int app_id, int customer_id, const Customer::RecvHandle& recv_handle)
    : app_id_(app_id), customer_id_(customer_id), recv_handle_(recv_handle) {
  customer_domain = new mxnet::profiler::ProfileDomain("CUSTOMER");
  accept_task = new mxnet::profiler::ProfileTask("ACCEPT", customer_domain);
  Postoffice::Get()->AddCustomer(this);
  recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
}

Customer::~Customer() {
  Postoffice::Get()->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  recv_thread_->join();
  delete accept_task;
  delete customer_domain;
}

int Customer::NewRequest(int recver) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  int num = Postoffice::Get()->GetNodeIDs(recver).size();
  tracker_.push_back(std::make_pair(num, 0));
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp].first == tracker_[timestamp].second;
    });
}

int Customer::NumResponse(int timestamp) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp].second;
}

void Customer::AddResponse(int timestamp, int num) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  tracker_[timestamp].second += num;
}

void Customer::Receiving() {
  while (true) {
    Message recv;
    recv_queue_.WaitAndPop(&recv);
    if (!recv.meta.control.empty() &&
        recv.meta.control.cmd == Control::TERMINATE) {
      break;
    }
    recv_handle_(recv);
    if (!recv.meta.request) {
      std::lock_guard<std::mutex> lk(tracker_mu_);
      tracker_[recv.meta.timestamp].second++;
      tracker_cond_.notify_all();
    }
  }
}

void Customer::Accept(const Message& recved) {
  accept_task->start();
  recv_queue_.Push(recved);
  // std::cout << "Data size: " + std::to_string(recved.data.size()) << std::endl;
  accept_task->stop();
  // if(Postoffice::Get()->is_server() && recved.meta.push) {
  // // construct response
  //   Message msg;
  //   msg.meta.app_id = app_id();
  //   msg.meta.customer_id = recved.meta.customer_id;
  //   msg.meta.request     = false;
  //   msg.meta.push        = recved.meta.push;
  //   msg.meta.head        = recved.meta.head;
  //   msg.meta.timestamp   = recved.meta.timestamp;
  //   msg.meta.recver      = recved.meta.sender;
  //   Postoffice::Get()->van()->Send(msg);
  // }
}

}  // namespace ps