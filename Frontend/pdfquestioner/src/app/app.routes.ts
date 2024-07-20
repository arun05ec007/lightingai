import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: 'chat',
    loadComponent: () =>
      import('./chat/chat.component').then((comp) => comp.ChatComponent),
  },
  {
    path: 'upload',
    loadComponent: () =>
      import('./upload/upload.component').then((comp) => comp.UploadComponent),
  },
  { path: '', pathMatch: 'full', redirectTo: 'chat' },
];
