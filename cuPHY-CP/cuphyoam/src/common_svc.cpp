/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <grpcpp/grpcpp.h>
#include <unistd.h>
#include <mutex>
#include <csignal>
#include "cuphyoam.hpp"
#include "oam_services.hpp"
#include "nv_ipc_forward.h"
#include <sstream>
#include <iomanip>
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_OAM + 4) // "OAM.COMMSVC"

#define RX_data_indication 0x85

std::mutex m;

grpc::Status CuphyCommonServiceImpl::WarmUp(grpc::ServerContext *context, const aerial::GenericRequest *request,
                                            aerial::DummyReply *reply)
{
    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::TerminateCuphycontroller(grpc::ServerContext* context, const aerial::GenericRequest* request,
                  aerial::DummyReply* reply)
{
    raise(SIGINT);
    return grpc::Status::OK;
}


grpc::Status CuphyCommonServiceImpl::GetSFN(grpc::ServerContext* context, const aerial::GenericRequest* request,
                                            aerial::SFNReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    uint32_t sfn_slot = oam->status_sfn_slot.load();
    reply->set_sfn(sfn_slot >> 8);
    reply->set_slot(sfn_slot & 0xFF);
    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::GetCpuUtilization(grpc::ServerContext* context, const aerial::GenericRequest* request,
                                                       aerial::CpuUtilizationReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    for (int k=0; k<MAX_CORES; k++)
    {
        if (oam->core_active[k].load())
        {
            aerial::CpuUtilizationPerCore *core = reply->add_core();
            core->set_core_id(k);
            core->set_utilization_x1000(oam->cpu_utilization_x1000[k].load());
        }
    }
    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SetPuschH5DumpNextCrc(grpc::ServerContext* context, const aerial::GenericRequest* request,
                                            aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    oam->puschH5dumpNextCrc.store(true);
    return grpc::Status::OK;
}


grpc::Status CuphyCommonServiceImpl::GetFAPIStream(grpc::ServerContext* context, const aerial::FAPIStreamRequest* request,
                  grpc::ServerWriter<aerial::FAPIStreamReply>* writer)
{
    const bool PRINT_DATA = false;
    bool status = true;

    int num_msgs = request->total_msgs_requested();

    nv_ipc_t* ipc = nv_ipc_get_instance(NULL);

    m.lock();
    if (this->active_clients == 0 && num_msgs==-1){
        nvipc_fw_start(ipc, 0);
    }
    else
        nvipc_fw_start(ipc, num_msgs);
    this->active_clients++;
    m.unlock();

    std::cout<<"Active Clients : "<<this->active_clients<<"\n";

    nv_ipc_msg_t msg;
    aerial::FAPIStreamReply reply;

    int dequeued_msg = 0;
    int msg_type = 0;

    while(status)
    {
        nvipc_fw_sem_wait(ipc); // BLOCKING CALL

        while(nvipc_fw_dequeue(ipc, &msg) >= 0)
        {

            // Set Client ID and populate Msg_buf
            reply.set_client_id(request->client_id());
            reply.set_msg_buf(reinterpret_cast<char*>(msg.msg_buf),msg.msg_len);


            // Copy data buffer only if type is RX_data.indication
            msg_type = (int)*(static_cast<uint16_t*>(msg.msg_buf)+1);

            if(msg_type == RX_data_indication)
                reply.set_data_buf(reinterpret_cast<char*>(msg.data_buf),msg.data_len);
            else
               reply.set_data_buf(nullptr, 0);

            // Free the IPC buffers
            nvipc_fw_free(ipc, &msg);

            status = writer->Write(reply);

            dequeued_msg++;

        }

        if(num_msgs != -1 && dequeued_msg >= num_msgs){
            break;
        }

    }

    m.lock();
    if (this->active_clients == 1 && num_msgs==-1){
        nvipc_fw_stop(ipc);
    }
    this->active_clients--;
    std::cout<<"Active Clients : "<<this->active_clients<<"\n";
    m.unlock();

    return grpc::Status::OK;
}

// datalake_save is a process-wide live gate, not a per-cell attribute.
// Returns true when the request was handled (including failure).
static bool handle_datalake_save(const aerial::CellParamUpdateRequest* request, grpc::Status* status, aerial::CellParamUpdateReply* reply)
{
    if (!request->multi_attrs_cfg() || request->attrs_size() != 1)
    {
        return false;
    }
    const auto it = request->attrs().find("datalake_save");
    if (it == request->attrs().end())
    {
        return false;
    }

    CuphyOAM* oam = CuphyOAM::getInstance();
    if (oam->callback.set_datalake_save == nullptr)
    {
        *status = grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
            "L1 has no datalake save hook; rebuild cuphycontroller");
        return true;
    }

    const int save_enable = it->second != 0.0 ? 1 : 0;
    if (oam->callback.set_datalake_save(save_enable) != 0)
    {
        *status = grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
            "datalake is not running; set data_config.data_core and restart cuphycontroller");
        return true;
    }

    if (reply != nullptr)
    {
        reply->mutable_resp()->set_status_code(aerial::Status::OK);
    }
    NVLOGC_FMT(TAG, "DataLake save {}", save_enable ? "on" : "off");
    *status = grpc::Status::OK;
    return true;
}

grpc::Status CuphyCommonServiceImpl::UpdateCellParamsSyncCall(grpc::ServerContext* context, const aerial::CellParamUpdateRequest* request,
                  aerial::CellParamUpdateReply* reply)
{
    grpc::Status save_status = grpc::Status::OK;
    if (handle_datalake_save(request, &save_status, reply))
    {
        return save_status;
    }

    CuphyOAM* oam = CuphyOAM::getInstance();

    // For update attenuation, update and return immediately (synchronous call).
    if (request->update_cell_attenuation_cfg() && oam->callback.update_cell_attenuation != nullptr)
    {
        if(oam->callback.update_cell_attenuation(request->cell_id(), request->attenuation_db()) != 0)
        {
            return grpc::Status::CANCELLED;
        }
    }

    if (request->update_network_cfg() || request->multi_attrs_cfg() || request->eaxcid_update())
    {
        auto new_cell_config = new CuphyOAMCellConfig();
        if (request->update_network_cfg())
        {
            new_cell_config->update_network_cfg = true;
            new_cell_config->cell_id = request->cell_id();
            new_cell_config->dst_mac_addr = request->dst_mac_addr();
            new_cell_config->vlan_tci = static_cast<uint16_t>(request->vlan_tci());
        }
        else if (request->multi_attrs_cfg())
        {
            new_cell_config->multi_attrs_cfg = true;
            new_cell_config->cell_id = request->cell_id();
            for (auto &p : request->attrs())
            {
                new_cell_config->attrs[p.first] = p.second;
            }
            oam->cell_multi_attri_update_callback(new_cell_config->cell_id, new_cell_config->attrs, new_cell_config->res);

            auto &res = new_cell_config->res;
            // Set the reply message
            if (res.size() == 0)
            {
                reply->mutable_resp()->set_status_code(aerial::Status::OK);
            }
            else
            {
                std::string key = "nic";

                uint64_t address = new_cell_config->attrs[key];
                uint32_t domain = (address >> 20) & 0xFFFF;
                uint32_t bus = (address >> 12) & 0xFF;
                uint32_t device = (address >> 4) & 0xFF;
                uint32_t function = address & 0xF;

                std::stringstream ss;
                ss << std::hex << std::setw(4) << std::setfill('0') << domain << ":"
                   << std::setw(2) << std::setfill('0') << bus << ":"
                   << std::setw(2) << std::setfill('0') << device << "."
                   << std::setw(1) << std::setfill('0') << function;

                std::string nic_pcie_addr = ss.str();

                reply->mutable_resp()->set_status_code(aerial::Status::ERROR_GENERAL);
                if (res.find(key) != res.end())
                {
                    std::string err_msg = res[key] == -1 ? ("Nic " + nic_pcie_addr + " not supported") : ("Nic  " + nic_pcie_addr + " setup failed");
                    reply->mutable_resp()->add_error_msgs(err_msg);
                }
                // reply->mutable_resp()->add_error_msgs("this is test error msg");
                // reply->mutable_resp()->add_error_msgs("this is 2nd test error msg");
            }
        }
        else if (request->eaxcid_update())
        {
            new_cell_config->eaxcid_update = true;
            new_cell_config->cell_id = request->cell_id();
            for (auto &p : request->ch_eaxcid_map())
            {
                auto &value = p.second;
                for (int i = 0; i < value.values_size(); ++i)
                {
                    new_cell_config->ch_eaxcid_map[p.first].push_back(value.values(i));
                }
            }
            oam->cell_eaxcids_update_callback(new_cell_config->cell_id, new_cell_config->ch_eaxcid_map);
        }

        delete new_cell_config;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::UpdateCellParams(grpc::ServerContext* context, const aerial::CellParamUpdateRequest* request,
                  aerial::DummyReply* reply)
{
    grpc::Status save_status = grpc::Status::OK;
    if (handle_datalake_save(request, &save_status, nullptr))
    {
        return save_status;
    }

    CuphyOAM* oam = CuphyOAM::getInstance();

    // For update attenuation, update and return immediately (synchronous call).
    if (request->update_cell_attenuation_cfg() && oam->callback.update_cell_attenuation != nullptr)
    {
        if(oam->callback.update_cell_attenuation(request->cell_id(), request->attenuation_db()) != 0)
        {
            return grpc::Status::CANCELLED;
        }
    }

    // For other configuration, enqueue the update request (asynchronous call).
    if (request->update_network_cfg() || request->multi_attrs_cfg())
    {
        auto new_cell_config = new CuphyOAMCellConfig();
        if (request->update_network_cfg())
        {
            new_cell_config->update_network_cfg = true;
            new_cell_config->cell_id = request->cell_id();
            new_cell_config->dst_mac_addr = request->dst_mac_addr();
            new_cell_config->vlan_tci = static_cast<uint16_t>(request->vlan_tci());
            NVLOGC_FMT(TAG, "Received gRPC msg to update network cfg: cell_id={}  mac_addr={} , vlan_tci={}",  new_cell_config->cell_id,  new_cell_config->dst_mac_addr,  new_cell_config->vlan_tci);
        }
        else if (request->multi_attrs_cfg())
        {
            new_cell_config->multi_attrs_cfg = true;
            new_cell_config->cell_id = request->cell_id();
            for(auto& p : request->attrs())
            {
                new_cell_config->attrs[p.first] = p.second;
                NVLOGC_FMT(TAG, "Received gRPC msg to update cell cfg: cell_id={}  key={} , value={}",  new_cell_config->cell_id,  p.first,  p.second);
            }
        }
        if (oam->put_cell_config(new_cell_config))
        {
            return grpc::Status::CANCELLED;
        }
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SendGenericAsyncCmd(grpc::ServerContext* context, const aerial::GenericAsyncRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    CuphyOAMGenericAsyncCmd* cmd = new CuphyOAMGenericAsyncCmd();
    cmd->cmd_id = request->cmd_id();
    cmd->param_int1 = request->param_int1();
    cmd->param_int2 = request->param_int2();
    cmd->param_str = request->param_str();
    if (oam->put_generic_async_cmd(cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SendCellCtrlCmd(grpc::ServerContext* context, const aerial::CellCtrlCmdRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    auto new_cell_ctrl_cmd = new CuphyOAMCellCtrlCmd();
    new_cell_ctrl_cmd->cell_id = request->cell_id();
    new_cell_ctrl_cmd->cell_ctrl_cmd = request->cell_ctrl_cmd();
    new_cell_ctrl_cmd->target_cell_id = request->target_cell_id();
    if (oam->put_cell_ctrl_cmd(new_cell_ctrl_cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SimulateCPUStall(grpc::ServerContext* context, const aerial::SimulatedCPUStallRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    auto simulated_cpu_stall_cmd = new SimulatedCPUStallCmd();
    simulated_cpu_stall_cmd->thread_id = request->thread_id();
    simulated_cpu_stall_cmd->task_id = request->task_id();
    simulated_cpu_stall_cmd->usleep_duration = request->usleep_duration();
    if (oam->put_simulated_cpu_stall_cmd(simulated_cpu_stall_cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SendFapiDelayCmd(grpc::ServerContext* context, const aerial::FapiDelayCmdRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    auto cmd = new CuphyOAMFapiDelayCmd();
    cmd->cell_id = request->cell_id();
    cmd->slot = request->slot();
    cmd->fapi_mask = request->fapi_mask();
    cmd->delay_us = request->delay_us();
    if (oam->put_fapi_delay_cmd(cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SimulateULUPlaneDrop(grpc::ServerContext* context, const aerial::SimulateULUPlaneDropRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    auto cmd = new CuphyOAMULUPlaneDropCmd();
    cmd->cell_id = request->cell_id();
    cmd->channel_id = request->channel_id();
    cmd->drop_rate = request->drop_rate();
    cmd->single_drop = request->single_drop();
    cmd->drop_slot = request->drop_slot();
    cmd->frame_id = request->frame_id();
    cmd->subframe_id = request->subframe_id();
    cmd->slot_id = request->slot_id();
    if (oam->put_ul_u_plane_drop_cmd(cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SendSfnSlotSyncCmd(grpc::ServerContext* context, const aerial::SfnSlotSyncRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();

    auto new_sfn_slot_sync_cmd = new CuphyOAMSfnSlotSyncCmd();
    new_sfn_slot_sync_cmd->sync_done = request->sync_done();
    new_sfn_slot_sync_cmd->time_anchor = request->time_anchor();
    if (oam->put_sfn_slot_sync_cmd(new_sfn_slot_sync_cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}                  

grpc::Status CuphyCommonServiceImpl::SendCellUlPcapCmd(grpc::ServerContext* context, const aerial::PcapRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    auto cell_id = request->cell_id();
    auto cmd = request->cmd();
    auto cell_mask = request->cell_mask();
    auto use_cell_mask = request->use_cell_mask();

    if(oam->ul_pcap_enabled.load())
    {
        if(use_cell_mask == 1)
        {
            auto prev = oam->ul_pcap_arm_cell_bitmask.load();
            oam->ul_pcap_arm_cell_bitmask.store(cell_mask);
            NVLOGC_FMT(TAG, "UL PCAP gRPC command received to update cell bitmask. Cell bitmask updated from {} to {}", prev, oam->ul_pcap_arm_cell_bitmask.load());
        }
        else
        {
            if(cell_id >= sizeof(uint64_t) * 8)
            {
                NVLOGC_FMT(TAG, "UL PCAP gRPC command received invalid cell id: {}...cell_ids accepted range [0,63]", cell_id);
                return grpc::Status::CANCELLED;
            }

            if(cell_id < 0)
            {
                NVLOGC_FMT(TAG, "UL PCAP gRPC command received invalid cell id: {}...cell_ids accepted range [0,63]", cell_id);
                return grpc::Status::CANCELLED;
            }

            if(cmd != 0 && cmd != 1)
            {
                NVLOGC_FMT(TAG, "UL PCAP gRPC command received invalid command: {}...commands accepted 0 - disable, 1 - enable", cmd);
                return grpc::Status::CANCELLED;
            }

            uint64_t mask = 1ULL << cell_id;
            uint64_t prev = 0;
            if (cmd) {
                prev = oam->ul_pcap_arm_cell_bitmask.fetch_or(mask, std::memory_order_relaxed);
            } else {
                prev = oam->ul_pcap_arm_cell_bitmask.fetch_and(~mask, std::memory_order_relaxed);
            }

            NVLOGC_FMT(TAG, "UL PCAP gRPC command received to {} cell {} for the next CRC error for the cell. Cell bitmask updated from {} to {}",
                (cmd == 0) ? "disarm" : "arm", cell_id, prev, oam->ul_pcap_arm_cell_bitmask.load());
        }
    }
    else
    {
        NVLOGC_FMT(TAG, "UL PCAP is not enabled, ignoring command");
        return grpc::Status::CANCELLED;
    }
    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::SendZeroUplane(grpc::ServerContext* context, const aerial::ZeroUplaneRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    auto cmd = new CuphyOAMZeroUplaneRequest();
    cmd->cell_id = request->cell_id();
    cmd->use_cell_mask = request->use_cell_mask();
    cmd->cell_mask = request->cell_mask();
    cmd->channel_id = request->channel_id();
    if (oam->put_zero_u_plane_request(cmd))
    {
        return grpc::Status::CANCELLED;
    }

    return grpc::Status::OK;
}

grpc::Status CuphyCommonServiceImpl::FlushUlPcap(grpc::ServerContext* context, const aerial::FlushUlPcapRequest* request,
                  aerial::DummyReply* reply)
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    auto cell_id = request->cell_id();
    if(cell_id >= sizeof(uint64_t) * 8)
    {
        NVLOGC_FMT(TAG, "UL PCAP gRPC command received invalid cell id: {}...cell_ids accepted range [0,63]", cell_id);
        return grpc::Status::CANCELLED;
    }

    if(cell_id < 0)
    {
        NVLOGC_FMT(TAG, "UL PCAP gRPC command received invalid cell id: {}...cell_ids accepted range [0,63]", cell_id);
        return grpc::Status::CANCELLED;
    }

    // If ul_pcap_arm_cell_bitmask is set for the cell, then flush the PCAP for the cell, else skip the flush
    uint64_t mask = 1ULL << cell_id;
    if(oam->ul_pcap_arm_cell_bitmask.load() & mask)
    {
        auto prev = oam->ul_pcap_flush_cell_bitmask.load();
        oam->ul_pcap_flush_cell_bitmask.fetch_or(mask, std::memory_order_relaxed);
        NVLOGC_FMT(TAG, "UL PCAP gRPC command received to flush PCAP for cell {}. Cell bitmask updated from {} to {}", cell_id, prev, oam->ul_pcap_flush_cell_bitmask.load());
        return grpc::Status::OK;
    }
    else
    {
        NVLOGC_FMT(TAG, "UL PCAP gRPC command received to flush PCAP for cell {}. Cell bitmask is not set, skipping flush", cell_id);
        return grpc::Status::CANCELLED;
    }
}
