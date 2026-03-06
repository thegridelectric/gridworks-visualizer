class AlertReactionRequest(BaseModel):
    house_alias: str
    new_status: str

self.app.post("/alert-reaction")(self.receive_alert_reaction)

async def receive_alert_reaction(self, alert_reaction: AlertReactionRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
        print(f"Received alert reaction: {alert_reaction}")
        if alert_reaction.new_status == 'ok':
            new_house_data = {
                "status": {'status': 'ok'},
            }
        else:
            return
        
        try:
            # Find the house by short_alias
            house = db.query(homes).filter(homes.short_alias == alert_reaction.house_alias).first()
            
            if not house:
                print(f"House '{alert_reaction.house_alias}' not found.")
                return False
            
            # Update the house with new data
            for key, value in new_house_data.items():
                if hasattr(house, key):
                    setattr(house, key, value)
            
            # Commit the changes
            db.commit()
            print(f"House '{alert_reaction.house_alias}' updated successfully")
            
        except Exception as e:
            print(f"Error updating house: {e}")