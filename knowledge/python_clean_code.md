# เทคนิคการเขียน Python (Design Patterns & Clean Code)

คู่มือฉบับนี้รวบรวมหลักการเขียน Python ให้เป็นแบบฉบับ Clean Code และการนำ Design Patterns มาปรับใช้ เพื่อให้โค้ดมีคุณภาพสูง อ่านง่าย และสามารถขยายระบบต่อได้ง่าย

## 1. หลักการตั้งชื่อ (Naming Conventions - PEP 8)
*   **Classes**: ให้ใช้ `CamelCase` (เช่น `DatabaseConnection`, `UserFactory`)
*   **Functions & Methods**: ให้ใช้ `snake_case` (เช่น `calculate_total`, `get_user_by_id`)
*   **Constants**: ให้ใช้ `UPPER_SNAKE_CASE` (เช่น `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
*   **Private/Protected Attributes**: ให้นำหน้าด้วย `_` (เช่น `_connection`, `_internal_cache`)

## 2. Type Hinting
บังคับให้ทุกฟังก์ชันและเมธอดต้องมีการระบุ Type Hinting (PEP 484) เสมอ เพื่อลดบั๊กและให้ IDE สามารถให้คำแนะนำได้

**❌ Bad:**
```python
def process_data(items):
    return [item.upper() for item in items]
```

**✅ Good:**
```python
from typing import List

def process_data(items: List[str]) -> List[str]:
    """แปลงรายชื่อเป็นตัวพิมพ์ใหญ่ทั้งหมด"""
    return [item.upper() for item in items]
```

## 3. Design Pattern: Singleton
เมื่อต้องการให้ระบบมี Object ของคลาสนั้นเพียงตัวเดียวเสมอ (เช่น Database Connection Pool หรือ Logger) ให้ใช้ Singleton Pattern โดยการ Overwrite `__new__`

```python
class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.log_file = "system.log"
            self._initialized = True

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")
```

## 4. Design Pattern: Factory Method
เมื่อระบบมีการสร้าง Object หลายประเภทที่มีคุณสมบัติคล้ายกัน ให้ใช้ Factory Method เพื่อลดความซ้ำซ้อนและทำให้เพิ่มคลาสใหม่ๆ ได้ง่าย

```python
from abc import ABC, abstractmethod

# 1. สร้าง Base Class (Interface)
class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> str:
        pass

# 2. สร้าง Concrete Classes
class EmailNotification(Notification):
    def send(self, message: str) -> str:
        return f"Sending Email: {message}"

class SMSNotification(Notification):
    def send(self, message: str) -> str:
        return f"Sending SMS: {message}"

# 3. สร้าง Factory Class
class NotificationFactory:
    @staticmethod
    def create_notification(noti_type: str) -> Notification:
        if noti_type.lower() == "email":
            return EmailNotification()
        elif noti_type.lower() == "sms":
            return SMSNotification()
        else:
            raise ValueError(f"Unknown notification type: {noti_type}")
```

## 5. การจัดการ Error (Exception Handling)
*   **อย่าใช้ `except Exception:` เปล่าๆ** ให้ระบุประเภทของ Error ให้ชัดเจนเสมอ
*   **ใช้ Custom Exceptions** เมื่อ Error นั้นเกี่ยวข้องกับ Business Logic ของแอปพลิเคชัน

```python
class InsufficientFundsError(Exception):
    """Exception raised for errors in the withdrawal process."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount > balance:
        raise InsufficientFundsError(f"Cannot withdraw {amount}, balance is only {balance}")
    return balance - amount
```

---
**เป้าหมาย:** AI ผู้ช่วยเขียนโค้ดต้องอ้างอิงและประยุกต์ใช้ Patterns เหล่านี้เสมอเมื่อถูกขอให้ "Refactor โค้ด", "สร้าง Class โครงสร้างใหม่" หรือ "เขียนฟังก์ชัน Python ที่ได้มาตรฐาน"
