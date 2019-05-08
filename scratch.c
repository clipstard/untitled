const int irsensorpin0 = A0;
const int irsensorpin1 = A1;
int flagFront = 0;
int flagBack = 0;
int timesSearch =0;
int momentFront = 0;
int momentBack= 0;
void setup() {
  
  // put your setup code here, to run once:
  Serial.begin(9600);}
void loop(){
  timesSearch +=1;
  // Reads the InfraRed sensor analog values and convert into distance.
  int sensorValue1 = analogRead(irsensorpin0);
  int sensorValue2 = analogRead(irsensorpin1);
  double IRdistance1 = 187754 * pow(sensorValue1, -1.51);
   double IRdistance2 = 187754 * pow(sensorValue2, -1.51);
   if (IRdistance1 <20){
     if (timesSearch - momentFront < 5){
     momentFront = timesSearch;
      flagFront +=1;
      if (flagFront ==6){
      flagFront =1;
      }
     } else {
     momentFront = timesSearch;
     flagFront = 1;
     }
   } else {
  flagFront = 0;
   }
   
    if (IRdistance2 <20){
     if (timesSearch - momentBack < 5){
     momentBack = timesSearch;
      flagBack +=1;
      if (flagBack ==6){
      flagBack =1;
      }
     } else {
     momentBack = timesSearch;
     flagBack = 1;
     }
   } else {
  flagBack = 0;
   }
 
   if (flagFront == 5){
   Serial.println("Obiect in fata"); 
 delay(5000);   
   } 
         
         
         if (flagBack == 5){
   Serial.println("Obiect in spate"); 
 delay(5000);   
   }                                            
    
     

  // A delay is added for a stable and precise input
  delay(500);
}
