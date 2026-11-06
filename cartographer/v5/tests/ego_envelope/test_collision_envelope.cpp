#include <plan_env/collision_envelope.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using plan_env::CollisionEnvelope;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(){
  try{
    const std::string good="EGO_ENVELOPE_V1 body-1 1 1000000000 1100000000 .10787 .10787 .12787";
    CollisionEnvelope off;
    require(off.ready(0,0)&&off.current(42,0,0),"default-off compatibility");
    CollisionEnvelope gate(true);
    require(!gate.ready(1000000000,1),"missing grant must block");
    require(gate.accept(good,1000000000,500),"valid atomic grant rejected");
    require(gate.ready(1000000000,500)&&gate.current(1,1000000000,500),"valid grant not usable");
    require(!gate.ready(1100000000,501),"ROS expiry boundary must block");
    require(!gate.ready(1000000000,100000500),"wall expiry with frozen ROS clock must block");
    require(!gate.ready(999999999,501),"backward ROS clock must block");
    require(!gate.ready(1000000000,499),"backward steady clock must block");
    CollisionEnvelope echoed(true);
    require(echoed.accept(gate.acceptedWire(),1000000000,500),"accepted ACK cannot roundtrip");
    require(echoed.record().sequence==1&&echoed.record().epoch=="body-1"&&
            echoed.record().extra==gate.record().extra,"ACK changed accepted fields");
    const std::vector<std::string> invalid={
      "", "EGO_ENVELOPE_V2 body-1 2 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body/1 2 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 0 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 -2 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 +2 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2.0 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 18446744073709551616 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 -1 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 9223372036854775808 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1000000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000001 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000001 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 900000000 1000000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 nan 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 inf 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 -.01 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0 1.00001 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0 0",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0 0 0 extra",
      "EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0xyz 0 0",
      "EGO_ENVELOPE_V1 "+std::string(129,'a')+" 2 1000000000 1100000000 0 0 0",
      std::string(513,'x'),good,
      "EGO_ENVELOPE_V1 body-2 2 1000000000 1100000000 0 0 0",
      "EGO_ENVELOPE_V1 body-1 2 999999999 1099999999 0 0 0"
    };
    for(const auto& wire:invalid){
      CollisionEnvelope g(true);require(g.accept(good,1000000000,500),"fixture setup");
      require(!g.accept(wire,1000000000,501),"invalid input accepted");
      require(!g.ready(1000000000,502),"invalid input left old grant active");
      require(g.record().sequence==1,"invalid packet advanced watermark");
      require(g.accept("EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0 0 0",1000000000,503),
              "later valid same-epoch input did not recover");
    }
    require(gate.accept("EGO_ENVELOPE_V1 body-1 2 1000000000 1100000000 0 .2 .3",1000000000,501),"refresh rejected");
    require(!gate.current(1,1000000000,502)&&gate.current(2,1000000000,502),"superseded plan token usable");
    require(!gate.accept("EGO_ENVELOPE_V1 body-1 1 1000000000 1100000000 0 0 0",1000000000,503),"old sequence accepted");
    CollisionEnvelope late(true);
    require(late.accept(good,1090000000,500),"late but valid packet rejected");
    require(late.ready(1090000000,10000499)&&!late.ready(1090000000,10000500),"receipt extended lifetime");
    std::cout<<"PASS: protocol, atomic ACK roundtrip, 26 invalid packets/recovery, supersession, strict ROS/steady deadlines\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
