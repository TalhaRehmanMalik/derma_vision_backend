from database import SessionLocal
import models

def seed_dermatologists():
    db = SessionLocal()
    try:
        count = db.query(models.Dermatologist).count()
        if count > 0:
            print("Dermatologists table already populated.")
            return

        doctors = [
            models.Dermatologist(
                name="Dr. Ayesha Khan",
                specialty="Consultant Dermatologist",
                hospital="Mayo Hospital / Skin Clinic",
                phone="+923001234567",
                availability="Mon - Sat (04:00 PM - 08:00 PM)",
                is_active=True
            ),
            models.Dermatologist(
                name="Dr. Usman Ahmed",
                specialty="Associate Professor Dermatology",
                hospital="General Hospital, Lahore",
                phone="+923219876543",
                availability="Mon - Fri (02:00 PM - 06:00 PM)",
                is_active=True
            )
        ]

        db.add_all(doctors)
        db.commit()
        print("Initial dermatologist data seeded successfully.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_dermatologists()