# คู่มือและสไตล์การเขียน React + Tailwind CSS (Enterprise UI Guidelines)

เอกสารนี้ระบุมาตรฐานการเขียนโค้ดสำหรับโปรเจกต์ Frontend ที่ใช้ React ร่วมกับ Tailwind CSS เพื่อให้โค้ดมีโครงสร้างที่เป็นระเบียบ บำรุงรักษาง่าย และมีดีไซน์ที่สม่ำเสมอในทุก Component

## 1. การตั้งชื่อ (Naming Conventions)
*   **Component Files**: ใช้ PascalCase (เช่น `UserProfile.jsx`, `PrimaryButton.jsx`)
*   **Functions/Variables**: ใช้ camelCase (เช่น `fetchUserData`, `isLoggedIn`)
*   **CSS Classes (Tailwind)**: ใช้รูปแบบ utility-first ปกติ แต่ถ้ามีการรวมกลุ่มให้ใช้การตัดบรรทัดให้เป็นระเบียบ

## 2. โครงสร้างของ React Component
ควรใช้ Functional Components ร่วมกับ Hooks เท่านั้น ไม่อนุญาตให้ใช้ Class Components

```jsx
import React, { useState, useEffect } from 'react';

const PrimaryButton = ({ label, onClick, disabled = false }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        px-6 py-2 rounded-lg font-semibold text-white transition-all duration-300
        ${disabled ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'}
      `}
    >
      {label}
    </button>
  );
};

export default PrimaryButton;
```

## 3. มาตรฐานสีขององค์กร (Brand Colors)
เมื่อสร้าง Component ใหม่ ให้ใช้ชุดสีขององค์กรดังนี้:
*   **Primary Color**: `bg-blue-600` (hover: `bg-blue-700`)
*   **Secondary Color**: `bg-gray-100` (hover: `bg-gray-200`, text: `text-gray-800`)
*   **Danger Color**: `bg-red-500` (hover: `bg-red-600`, text: `text-white`)
*   **Success Color**: `bg-green-500` (hover: `bg-green-600`, text: `text-white`)

## 4. โครงสร้าง Layout พื้นฐาน (Page Wrapper)
ทุกหน้า Page ควรถูกครอบด้วย Wrapper ที่กำหนด padding และ background เพื่อความสม่ำเสมอ:

```jsx
const PageWrapper = ({ children, title }) => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-10 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-4xl bg-white shadow-md rounded-xl p-8">
        {title && <h1 className="text-3xl font-bold text-gray-900 mb-6">{title}</h1>}
        <div className="text-gray-700">
          {children}
        </div>
      </div>
    </div>
  );
};
```

## 5. การจัดการ Form และ Input
Input fields ทั้งหมดต้องใช้รูปแบบเดียวกัน โดยให้มี Border สีเทา, Rounded corners แบบ `rounded-md`, และมี Focus ring สี Blue:

```jsx
const TextInput = ({ label, placeholder, value, onChange }) => {
  return (
    <div className="flex flex-col space-y-2 mb-4">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        className="px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow"
      />
    </div>
  );
};
```

---
**หมายเหตุ:** ให้ยึดคู่มือฉบับนี้เสมอเมื่อมีผู้ใช้สั่งให้ "สร้าง Component", "สร้างหน้า UI" หรือ "ออกแบบฟอร์ม" ด้วย React และ Tailwind CSS
